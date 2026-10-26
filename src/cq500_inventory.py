"""Inventory every DICOM series in CQ500 (headers only, no pixel data).

Each CQ500 scan folder holds several series (thin/thick, soft/bone kernel, sometimes
contrast or reformats). Before choosing ONE series per scan we need to know what exists.

Outputs (data/headct/):
  series_inventory.csv   one row per series: thickness, z-spacing, kernel, tilt, shear, ...
  file_index.csv.gz      one row per DICOM file: case, series, SOPInstanceUID, z-position, path
                         (needed later to place the BHX boxes, which are keyed by SOPInstanceUID)

Run:  python src/cq500_inventory.py --workers 8
      (header-only reading of ~193k files: roughly 5-15 min on a laptop SSD)
"""
import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from config import CQ500_DICOM_DIR, HEADCT_DIR
from cq500_labels import normalise_case_id

TAGS = ["SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID", "SeriesNumber", "SeriesDescription",
        "InstanceNumber", "Modality", "ImageType", "ConvolutionKernel", "SliceThickness", "PixelSpacing",
        "Rows", "Columns", "ImagePositionPatient", "ImageOrientationPatient", "GantryDetectorTilt",
        "ContrastBolusAgent", "Manufacturer", "ManufacturerModelName"]


# --------------------------------------------------------------------------- #
# Reading (the only part that needs pydicom)
# --------------------------------------------------------------------------- #
def _to_py(v):
    """pydicom values -> plain python: lists for multi-valued, float for numbers, str otherwise."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
        return [_to_py(x) for x in v]
    if isinstance(v, (int, float)):          # DSfloat / IS are int/float subclasses
        return float(v)
    return str(v)


def read_header(path):
    """Header of one DICOM file as a dict (None if it is not a readable DICOM)."""
    import pydicom  # imported here so the pure-pandas parts can be tested without pydicom
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=TAGS)
    except Exception:  # noqa: BLE001  (not DICOM / truncated)
        return None
    if "SeriesInstanceUID" not in ds:
        return None
    h = {t: _to_py(ds.get(t)) for t in TAGS}
    h["path"] = path
    return h


def _read_many(paths):
    return [read_header(p) for p in paths]


def list_files(root):
    files = []
    for d, _, names in os.walk(root):
        files += [os.path.join(d, n) for n in names if not n.startswith(".") and
                  (n.lower().endswith(".dcm") or "." not in n)]
    return sorted(files)


# --------------------------------------------------------------------------- #
# Geometry / summarising (pure python, unit-tested)
# --------------------------------------------------------------------------- #
def slice_normal(iop):
    """Unit normal of the image plane from ImageOrientationPatient (6 direction cosines)."""
    r, c = np.asarray(iop[:3], float), np.asarray(iop[3:6], float)
    n = np.cross(r, c)
    return n / np.linalg.norm(n)


def _angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    cos = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _first(values):
    vals = [v for v in values if v not in (None, "", [])]
    return vals[0] if vals else None


def _join(values):
    vals = sorted({("\\".join(map(str, v)) if isinstance(v, list) else str(v)) for v in values
                   if v not in (None, "", [])})
    return "|".join(vals)


def _median(values):
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return float(np.median(vals)) if vals else np.nan


def summarise_series(headers):
    """One series (list of header dicts) -> one inventory row."""
    iop = _first(h["ImageOrientationPatient"] for h in headers)
    ipps = [h["ImagePositionPatient"] for h in headers if h.get("ImagePositionPatient")]
    row = {
        "series_uid": headers[0]["SeriesInstanceUID"],
        "study_uid": headers[0]["StudyInstanceUID"],
        "series_number": _first(h["SeriesNumber"] for h in headers),
        "series_description": _join(h["SeriesDescription"] for h in headers),
        "series_dir": Counter(os.path.dirname(h["path"]) for h in headers).most_common(1)[0][0],
        "n_files": len(headers),
        "modality": _join(h["Modality"] for h in headers),
        "image_type": _join(h["ImageType"] for h in headers),
        "kernel": _join(h["ConvolutionKernel"] for h in headers),
        "slice_thickness": _median([h.get("SliceThickness") for h in headers]),
        "pixel_spacing": _median([(h.get("PixelSpacing") or [None])[0] for h in headers]),
        "rows": _first(h["Rows"] for h in headers),
        "cols": _first(h["Columns"] for h in headers),
        "gantry_tilt_hdr": _first(h["GantryDetectorTilt"] for h in headers),
        "contrast_agent": _join(h["ContrastBolusAgent"] for h in headers),
        "manufacturer": _join(h["Manufacturer"] for h in headers),
        "model": _join(h["ManufacturerModelName"] for h in headers),
        "n_orientations": len({tuple(np.round(h["ImageOrientationPatient"], 3))
                               for h in headers if h.get("ImageOrientationPatient")}),
    }
    row.update(dict(plane_tilt_deg=np.nan, shear_deg=np.nan, z_spacing=np.nan, z_spacing_cv=np.nan,
                    n_positions=0, z_extent_mm=np.nan))
    if iop is not None:
        n = slice_normal(iop)
        row["plane_tilt_deg"] = _angle_deg(n, [0, 0, 1])          # 0 = pure axial
        if len(ipps) >= 3:
            P = np.asarray(ipps, float)
            z = P @ n
            order = np.argsort(z)
            zs, Ps = z[order], P[order]
            uz = np.unique(np.round(zs, 3))
            row["n_positions"] = int(len(uz))
            row["z_extent_mm"] = float(uz[-1] - uz[0])
            if len(uz) >= 2:
                d = np.diff(uz)
                row["z_spacing"] = float(np.median(d))
                row["z_spacing_cv"] = float(np.std(d) / np.mean(d))
                # shear: slice-to-slice displacement not parallel to the plane normal
                # (gantry tilt stored in positions, not orientation) -> needs resampling later
                step = Ps[-1] - Ps[0]
                if np.linalg.norm(step) > 0:
                    row["shear_deg"] = _angle_deg(step, n)
    return row


def _zpos(h):
    if not h.get("ImagePositionPatient") or not h.get("ImageOrientationPatient"):
        return np.nan
    return float(np.asarray(h["ImagePositionPatient"], float) @ slice_normal(h["ImageOrientationPatient"]))


def build_inventory(headers):
    """All header dicts -> (series_inventory, file_index). Works on plain dicts (not a DataFrame),
    so missing tags stay None instead of NaN."""
    headers = [h for h in headers if h]
    if not headers:
        raise RuntimeError("No DICOM headers could be read - check --dicom_dir.")
    groups = {}
    for h in headers:
        h["case_id"] = normalise_case_id(h["path"])
        groups.setdefault((h["case_id"] or "", h["SeriesInstanceUID"]), []).append(h)

    rows = []
    for (case_id, _), g in sorted(groups.items()):
        r = {"case_id": case_id or None}
        r.update(summarise_series(g))
        rows.append(r)
    inv = pd.DataFrame(rows)

    files = pd.DataFrame([{"case_id": h["case_id"], "series_uid": h["SeriesInstanceUID"],
                           "sop_uid": h["SOPInstanceUID"], "instance_number": h["InstanceNumber"],
                           "z": _zpos(h), "path": h["path"]} for h in headers])
    return inv, files


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dicom_dir", default=str(CQ500_DICOM_DIR))
    p.add_argument("--out_dir", default=str(HEADCT_DIR))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit_cases", type=int, default=None, help="Only the first N scan folders (quick test).")
    args = p.parse_args()

    files = list_files(args.dicom_dir)
    if args.limit_cases:
        keep = sorted({normalise_case_id(f) for f in files} - {None},
                      key=lambda c: int(c.split("-")[-1]))[: args.limit_cases]
        files = [f for f in files if normalise_case_id(f) in keep]
    print(f"{len(files)} candidate files under {args.dicom_dir}")

    chunks = [files[i:i + 500] for i in range(0, len(files), 500)]
    headers = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, part in enumerate(ex.map(_read_many, chunks), 1):
            headers += part
            if i % 20 == 0 or i == len(chunks):
                print(f"  read {min(i * 500, len(files))}/{len(files)}")
    n_bad = sum(h is None for h in headers)
    inv, file_index = build_inventory(headers)
    missing_case = inv["case_id"].isna().sum()

    os.makedirs(args.out_dir, exist_ok=True)
    inv.to_csv(os.path.join(args.out_dir, "series_inventory.csv"), index=False)
    file_index.to_csv(os.path.join(args.out_dir, "file_index.csv.gz"), index=False)

    print(f"\nUnreadable/non-DICOM files: {n_bad}   series without a parsable case id: {missing_case}")
    print(f"{inv['case_id'].nunique()} scans, {len(inv)} series "
          f"({len(inv) / max(inv['case_id'].nunique(), 1):.1f} per scan)")
    print("\nSlice thickness (mm) x number of series:")
    print(inv["slice_thickness"].round(2).value_counts().sort_index().to_string())
    print("\nKernels:")
    print(inv["kernel"].value_counts().head(25).to_string())
    print("\nMost common series descriptions:")
    print(inv["series_description"].value_counts().head(25).to_string())
    print(f"\nSeries with plane tilt > 1 deg: {(inv['plane_tilt_deg'] > 1).sum()}   "
          f"with shear > 1 deg: {(inv['shear_deg'] > 1).sum()}")
    print(f"\nSaved -> {args.out_dir}/series_inventory.csv, file_index.csv.gz")


if __name__ == "__main__":
    main()
