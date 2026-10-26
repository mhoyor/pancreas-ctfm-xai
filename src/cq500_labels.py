"""Build the CQ500 label table from the three radiologists' reads.

Input : $CQ500_ROOT/reads.csv  (columns: name, Category, R1:ICH, R1:IPH, ..., R2:..., R3:...)
Output: data/headct/labels.csv           one row per scan
        data/headct/label_agreement.csv  Fleiss' kappa per finding (report this in Methods)

Primary label = ICH by MAJORITY VOTE (>= 2 of 3 readers), as in the original CQ500 paper.
CT-FM's own CQ500 evaluation used "any reader, any finding" (abnormal_any_reader); it is kept
as a column so the CT-FM setup can be reproduced as a secondary analysis.

Run: python src/cq500_labels.py
"""
import argparse
import math
import re

import numpy as np
import pandas as pd

from config import CQ500_READS, HEADCT_DIR

READER_COL = re.compile(r"^(R\d+):(.+)$")
CASE_ID = re.compile(r"CQ500[-_ ]?CT[-_ ]?(\d+)", re.IGNORECASE)


def normalise_case_id(name):
    """'CQ500-CT-12', 'CQ500CT12', 'cq500_ct_12' -> 'CQ500-CT-12'. Returns None if no match."""
    m = CASE_ID.search(str(name))
    return f"CQ500-CT-{int(m.group(1))}" if m else None


def fleiss_kappa_binary(votes, n_readers):
    """Fleiss' kappa for a binary finding. `votes` = number of readers saying 1, per scan."""
    votes = np.asarray(votes, dtype=float)
    counts = np.stack([n_readers - votes, votes], axis=1)            # (N, 2) category counts
    p_j = counts.sum(axis=0) / counts.sum()                          # overall category shares
    p_i = ((counts ** 2).sum(axis=1) - n_readers) / (n_readers * (n_readers - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    return float("nan") if math.isclose(p_e, 1.0) else float((p_bar - p_e) / (1 - p_e))


def build_labels(reads):
    """Turn the wide reads table into one row per scan with vote counts and majority labels."""
    reader_cols = {c: READER_COL.match(c).groups() for c in reads.columns if READER_COL.match(c)}
    if not reader_cols:
        raise ValueError(f"No reader columns like 'R1:ICH' found. Columns: {list(reads.columns)}")
    readers = sorted({r for r, _ in reader_cols.values()})
    findings = sorted({f for _, f in reader_cols.values()})
    n = len(readers)
    majority = n // 2 + 1  # strict majority: 2 of 3

    out = pd.DataFrame({"case_id": reads["name"].map(normalise_case_id)})
    if out["case_id"].isna().any():
        bad = reads.loc[out["case_id"].isna(), "name"].tolist()
        raise ValueError(f"Could not parse case ids: {bad[:10]}")
    if "Category" in reads.columns:
        out["category"] = reads["Category"].astype(str).values

    agreement = []
    for f in findings:
        cols = [f"{r}:{f}" for r in readers if f"{r}:{f}" in reads.columns]
        v = reads[cols].fillna(0).astype(int).clip(0, 1).sum(axis=1).values
        out[f"{f}_votes"] = v
        out[f"{f}"] = (v >= majority).astype(int)
        agreement.append({"finding": f, "n_positive_majority": int((v >= majority).sum()),
                          "n_unanimous_positive": int((v == len(cols)).sum()),
                          "fleiss_kappa": round(fleiss_kappa_binary(v, len(cols)), 3)})

    vote_cols = [f"{f}_votes" for f in findings]
    out["abnormal_any_reader"] = (out[vote_cols].sum(axis=1) > 0).astype(int)   # CT-FM definition
    out["abnormal_majority"] = (out[findings].sum(axis=1) > 0).astype(int)
    if "ICH" in findings:
        out["ICH_unanimous"] = out["ICH_votes"].isin([0, len(readers)]).astype(int)
    return out, pd.DataFrame(agreement), readers


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reads", default=str(CQ500_READS))
    p.add_argument("--out_dir", default=str(HEADCT_DIR))
    args = p.parse_args()

    reads = pd.read_csv(args.reads)
    labels, agreement, readers = build_labels(reads)
    HEADCT_DIR.mkdir(parents=True, exist_ok=True)
    labels.to_csv(f"{args.out_dir}/labels.csv", index=False)
    agreement.to_csv(f"{args.out_dir}/label_agreement.csv", index=False)

    print(f"{len(labels)} scans, readers: {readers}")
    print(f"ICH (majority): {labels['ICH'].sum()} positive / {(1 - labels['ICH']).sum()} negative")
    print(f"ICH unanimous: {labels['ICH_unanimous'].sum()} of {len(labels)} scans")
    print("\nReader agreement (Fleiss' kappa):")
    print(agreement.to_string(index=False))
    if "category" in labels:
        print("\nCategory x ICH  (check the batch confound: B1 vs B2 prevalence):")
        print(pd.crosstab(labels["category"], labels["ICH"], margins=True))
    print(f"\nSaved -> {args.out_dir}/labels.csv, label_agreement.csv")


if __name__ == "__main__":
    main()
