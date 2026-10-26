"""Choose ONE series per CQ500 scan, using the same rules for every scan.

Why the rules matter: if bleed-positive scans systematically got a different kind of series
(thinner slices, another kernel) than negatives, the classifier could learn the series type
instead of the bleed. So eligibility and ranking are identical for all scans; the only
label-dependent step is a tie-breaker that prefers the series the BHX neuroradiologists drew
their boxes on, and the script prints crosstabs so you can check that nothing leaks.

Eligible series (defaults follow CT-FM's head CT setup, which resamples to 5 mm slices):
  CT, axial (plane tilt <= --max_tilt), single orientation, not a localizer,
  soft-tissue / brain kernel (not bone/sharp), no contrast,
  slice thickness in [--min_thk, --max_thk] mm, >= --min_slices slices,
  covers >= --min_extent mm, regular slice spacing.
If a scan has no eligible thick series, the thin-slice fallback (thickness < --min_thk) is used
and flagged (`fallback_thin`), so you can exclude or analyse those scans separately.

Ranking among eligible series: BHX boxes present > |thickness - 5 mm| > ORIGINAL over DERIVED
> more slices > lower series number.

Inputs : data/headct/series_inventory.csv, data/headct/labels.csv, BHX manual csv (optional)
Outputs: data/headct/selected_series.csv, data/headct/selection_excluded.csv

Run: python src/cq500_select_series.py
"""
import argparse
import ast
import json
import os
import re

import numpy as np
import pandas as pd

from config import BHX_MANUAL_CSV, HEADCT_DIR

# Bone/sharp kernels: GE names + Siemens H/B/J kernels >= 60. Vendor naming varies, so CHECK the
# kernel list printed by cq500_inventory.py and extend this pattern if a bone kernel slips through.
BONE_KERNEL = re.compile(r"BONE|SHARP|EDGE|LUNG|DETAIL|\bH[6-9]\d|\bB[6-9]\d|\bJ[6-9]\d", re.IGNORECASE)
BONE_DESC = re.compile(r"BONE|SHARP|LUNG", re.IGNORECASE)
# Post-contrast series only. "PRE CONTRAST" / "NON CONTRAST" / "WITHOUT CONTRAST" mean the opposite and are
# removed first (CQ500 names many plain scans "PRE CONTRAST 5MM STD"); bare "POST" is not used because of
# descriptions like "POST FOSSA" (posterior fossa).
NON_CONTRAST_DESC = re.compile(r"\b(?:PRE|NON|WITHOUT|W/?O)[\s_-]*(?:CONTRAST|CON|C)\b|\bPLAIN\b", re.IGNORECASE)
CONTRAST_DESC = re.compile(r"C\+|\bCONTRAST\b|\bCTA\b|ANGIO|\bCE\b|\bWITH[\s_-]*(?:CONTRAST|C)\b", re.IGNORECASE)
LOCALIZER = re.compile(r"LOCALIZER|SCOUT|TOPOGRAM|SURVIEW", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# BHX
# --------------------------------------------------------------------------- #
def parse_box(value):
    """BHX 'data' column -> dict(x, y, width, height) in pixel units. Accepts JSON or python-dict text."""
    if isinstance(value, dict):
        d = value
    else:
        text = str(value)
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            d = ast.literal_eval(text)
    return {k: float(d[k]) for k in ("x", "y", "width", "height")}


def load_bhx_counts(path):
    """Number of manual BHX boxes per SeriesInstanceUID (empty frame if the file is missing)."""
    if not path or not os.path.exists(path):
        print(f"NOTE: BHX file not found ({path}); selecting without BHX tie-breaker.")
        return pd.DataFrame(columns=["series_uid", "bhx_boxes", "bhx_images"])
    bhx = pd.read_csv(path)
    return (bhx.groupby("SeriesInstanceUID")
               .agg(bhx_boxes=("SOPInstanceUID", "size"), bhx_images=("SOPInstanceUID", "nunique"))
               .reset_index().rename(columns={"SeriesInstanceUID": "series_uid"}))


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def flag_series(inv, min_thk=2.5, max_thk=6.0, max_tilt=30.0, min_slices=15, min_extent=80.0,
                max_spacing_cv=0.2):
    """Add boolean rule columns + `eligible_thick` / `eligible_thin` to the inventory."""
    s = inv.copy()
    txt = lambda c: s[c].fillna("").astype(str)
    s["is_ct"] = txt("modality").str.contains("CT")
    s["is_localizer"] = txt("image_type").str.contains(LOCALIZER) | txt("series_description").str.contains(LOCALIZER)
    s["is_axial"] = (s["plane_tilt_deg"] <= max_tilt) & (s["n_orientations"] == 1)
    s["is_bone"] = txt("kernel").str.contains(BONE_KERNEL) | txt("series_description").str.contains(BONE_DESC)
    desc_post = txt("series_description").str.replace(NON_CONTRAST_DESC, " ", regex=True)
    s["is_contrast"] = (txt("contrast_agent").str.strip() != "") | desc_post.str.contains(CONTRAST_DESC)
    s["enough_slices"] = s["n_positions"] >= min_slices
    s["covers_head"] = s["z_extent_mm"] >= min_extent
    s["regular_spacing"] = s["z_spacing_cv"].fillna(1.0) <= max_spacing_cv
    s["is_original"] = txt("image_type").str.upper().str.startswith("ORIGINAL")
    thk = s["slice_thickness"].fillna(s["z_spacing"])
    s["thk_used"] = thk
    base = (s["is_ct"] & ~s["is_localizer"] & s["is_axial"] & ~s["is_bone"] & ~s["is_contrast"]
            & s["covers_head"] & s["regular_spacing"])
    s["eligible_thick"] = base & thk.between(min_thk, max_thk) & s["enough_slices"]
    s["eligible_thin"] = base & (thk < min_thk) & s["enough_slices"]
    return s


def select(inv, labels, bhx_counts, **rule_kwargs):
    s = flag_series(inv, **rule_kwargs)
    s = s.merge(bhx_counts, on="series_uid", how="left")
    s[["bhx_boxes", "bhx_images"]] = s[["bhx_boxes", "bhx_images"]].fillna(0).astype(int)
    s["thk_dist"] = (s["thk_used"] - 5.0).abs()

    chosen, excluded = [], []
    for case_id, g in s.groupby("case_id", sort=False):
        pool, fallback = g[g["eligible_thick"]], False
        if pool.empty:
            pool, fallback = g[g["eligible_thin"]], True
        if pool.empty:
            excluded.append({"case_id": case_id, "n_series": len(g),
                             "reason": "no eligible soft-kernel non-contrast axial series"})
            continue
        pool = pool.assign(_has_bhx=(pool["bhx_boxes"] > 0).astype(int)).sort_values(
            ["_has_bhx", "thk_dist", "is_original", "n_positions", "series_number"],
            ascending=[False, True, False, False, True])
        best = pool.iloc[0].drop(labels=["_has_bhx"]).to_dict()
        best["fallback_thin"] = fallback
        best["n_eligible"] = len(pool)
        best["bhx_boxes_in_scan"] = int(g["bhx_boxes"].sum())   # boxes on ANY series of this scan
        chosen.append(best)

    sel = pd.DataFrame(chosen)
    lab_cols = [c for c in ["case_id", "category", "ICH", "ICH_votes", "ICH_unanimous", "abnormal_any_reader",
                            "IPH", "IVH", "SDH", "EDH", "SAH", "ChronicBleed"] if c in labels.columns]
    sel = labels[lab_cols].merge(sel, on="case_id", how="left", indicator=True)
    rule_excluded = {e["case_id"] for e in excluded}
    no_series = [c for c in sel.loc[sel["_merge"] == "left_only", "case_id"] if c not in rule_excluded]
    sel = sel[sel["_merge"] == "both"].drop(columns="_merge")
    excl = pd.DataFrame(excluded)
    if no_series:
        excl = pd.concat([excl, pd.DataFrame({"case_id": no_series, "reason": "in labels but no DICOM found"})])
    keep = ["case_id", "category", "ICH", "ICH_votes", "ICH_unanimous", "abnormal_any_reader",
            "series_uid", "series_number", "series_description", "series_dir", "thk_used", "z_spacing",
            "n_positions", "kernel", "manufacturer", "model", "plane_tilt_deg", "shear_deg", "is_original",
            "fallback_thin", "n_eligible", "bhx_boxes", "bhx_images", "bhx_boxes_in_scan"]
    sel = sel[[c for c in keep if c in sel.columns] +
              [c for c in ["IPH", "IVH", "SDH", "EDH", "SAH", "ChronicBleed"] if c in sel.columns]]
    return sel.reset_index(drop=True), excl


def report(sel, excl):
    print(f"Selected {len(sel)} scans; excluded {len(excl)}")
    if len(excl):
        print(excl["reason"].value_counts().to_string())
    print("\nICH x thin-slice fallback:")
    print(pd.crosstab(sel["ICH"], sel["fallback_thin"], margins=True))
    print("\nICH x slice thickness of chosen series:")
    print(pd.crosstab(sel["thk_used"].round(1), sel["ICH"], margins=True))
    print("\nICH x manufacturer:")
    print(pd.crosstab(sel["manufacturer"], sel["ICH"], margins=True))
    if "category" in sel:
        print("\nICH x category (batch):")
        print(pd.crosstab(sel["category"], sel["ICH"], margins=True))
    pos = sel["ICH"] == 1
    print(f"\nICH-positive scans: {pos.sum()}  | chosen series has BHX boxes: {(pos & (sel['bhx_boxes'] > 0)).sum()}"
          f"  | boxes only on another series: {(pos & (sel['bhx_boxes'] == 0) & (sel['bhx_boxes_in_scan'] > 0)).sum()}"
          f"  | no boxes at all: {(pos & (sel['bhx_boxes_in_scan'] == 0)).sum()}")
    neg_box = (~pos) & (sel["bhx_boxes_in_scan"] > 0)
    print(f"ICH-negative scans that nevertheless have BHX boxes (minority-reader bleeds; check): {neg_box.sum()}")
    print(f"Chosen series with shear > 1 deg (need resampling at conversion): {(sel['shear_deg'] > 1).sum()}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inventory", default=str(HEADCT_DIR / "series_inventory.csv"))
    p.add_argument("--labels", default=str(HEADCT_DIR / "labels.csv"))
    p.add_argument("--bhx", default=str(BHX_MANUAL_CSV))
    p.add_argument("--out_dir", default=str(HEADCT_DIR))
    p.add_argument("--min_thk", type=float, default=2.5)
    p.add_argument("--max_thk", type=float, default=6.0)
    p.add_argument("--max_tilt", type=float, default=30.0)
    p.add_argument("--min_slices", type=int, default=15)
    p.add_argument("--min_extent", type=float, default=80.0)
    args = p.parse_args()

    inv, labels = pd.read_csv(args.inventory), pd.read_csv(args.labels)
    sel, excl = select(inv, labels, load_bhx_counts(args.bhx), min_thk=args.min_thk, max_thk=args.max_thk,
                       max_tilt=args.max_tilt, min_slices=args.min_slices, min_extent=args.min_extent)
    sel.to_csv(os.path.join(args.out_dir, "selected_series.csv"), index=False)
    excl.to_csv(os.path.join(args.out_dir, "selection_excluded.csv"), index=False)
    report(sel, excl)
    print(f"\nSaved -> {args.out_dir}/selected_series.csv, selection_excluded.csv")


if __name__ == "__main__":
    main()
