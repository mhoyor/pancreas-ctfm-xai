"""Synthetic checks for the CQ500 label / inventory / selection logic (no real data, no pydicom needed).

Run: python tests/test_cq500_data.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cq500_inventory import build_inventory, summarise_series  # noqa: E402
from cq500_labels import build_labels, fleiss_kappa_binary, normalise_case_id  # noqa: E402
from cq500_select_series import parse_box, select  # noqa: E402

AXIAL = [1, 0, 0, 0, 1, 0]


def fake_series(case, uid, n, thk, kernel="STANDARD", desc="CT PLAIN", tilt_deg=0.0, shear=False,
                image_type=("ORIGINAL", "PRIMARY", "AXIAL"), contrast=None, start=0):
    """Header dicts for one synthetic series. Tilt goes into the orientation (GE style) or,
    with shear=True, only into the positions (sheared stack)."""
    t = np.radians(tilt_deg)
    iop = [1, 0, 0, 0, np.cos(t), np.sin(t)] if not shear else AXIAL
    normal = np.cross(iop[:3], iop[3:])
    out = []
    for i in range(n):
        pos = (np.array([0, i * thk * np.tan(np.radians(15)), 0]) if shear else np.zeros(3)) + normal * (i * thk)
        out.append({
            "SOPInstanceUID": f"{uid}.{i}", "SeriesInstanceUID": uid, "StudyInstanceUID": f"st{case}",
            "SeriesNumber": float(start), "SeriesDescription": desc, "InstanceNumber": float(i + 1),
            "Modality": "CT", "ImageType": list(image_type), "ConvolutionKernel": kernel,
            "SliceThickness": float(thk), "PixelSpacing": [0.45, 0.45], "Rows": 512.0, "Columns": 512.0,
            "ImagePositionPatient": [float(x) for x in pos], "ImageOrientationPatient": [float(x) for x in iop],
            "GantryDetectorTilt": float(tilt_deg), "ContrastBolusAgent": contrast, "Manufacturer": "GE",
            "ManufacturerModelName": "X",
            "path": f"/d/CQ500CT{case} CQ500CT{case}/Unknown Study/{desc}/CT{i:06d}.dcm",
        })
    return out


def test_case_ids():
    assert normalise_case_id("CQ500-CT-12") == "CQ500-CT-12"
    assert normalise_case_id("/x/CQ500CT7 CQ500CT7/Unknown Study/a.dcm") == "CQ500-CT-7"
    assert normalise_case_id("nothing here") is None


def test_labels_majority_and_kappa():
    reads = pd.DataFrame({
        "name": ["CQ500-CT-0", "CQ500-CT-1", "CQ500-CT-2", "CQ500-CT-3"],
        "Category": ["B1", "B1", "B2", "B2"],
        "R1:ICH": [1, 1, 0, 0], "R2:ICH": [1, 0, 0, 1], "R3:ICH": [1, 0, 0, 1],
        "R1:Fracture": [0, 0, 1, 0], "R2:Fracture": [0, 0, 0, 0], "R3:Fracture": [0, 0, 0, 0],
    })
    lab, agr, readers = build_labels(reads)
    assert readers == ["R1", "R2", "R3"]
    assert lab["ICH"].tolist() == [1, 0, 0, 1]                 # 3/3, 1/3, 0/3, 2/3
    assert lab["ICH_unanimous"].tolist() == [1, 0, 1, 0]
    assert lab["abnormal_any_reader"].tolist() == [1, 1, 1, 1]   # CT-FM definition: any reader, any finding
    assert lab["abnormal_majority"].tolist() == [1, 0, 0, 1]
    assert fleiss_kappa_binary([3, 3, 0, 0], 3) == 1.0          # perfect agreement
    assert fleiss_kappa_binary([1, 2, 1, 2], 3) < 0             # worse than chance


def test_series_geometry():
    r = summarise_series(fake_series(0, "a", 30, 5.0))
    assert r["n_positions"] == 30 and abs(r["z_spacing"] - 5.0) < 1e-6 and r["plane_tilt_deg"] < 1e-6
    assert abs(r["z_extent_mm"] - 145.0) < 1e-6
    r = summarise_series(fake_series(0, "b", 30, 5.0, tilt_deg=20))   # tilt in orientation: fine, no shear
    assert abs(r["plane_tilt_deg"] - 20) < 1e-3 and r["shear_deg"] < 1e-3
    r = summarise_series(fake_series(0, "c", 30, 5.0, shear=True))    # tilt only in positions: sheared
    assert abs(r["shear_deg"] - 15) < 1e-3


def test_inventory_and_selection():
    headers = (fake_series(0, "s0_thin", 200, 0.625, desc="CT PLAIN THIN", start=2)
               + fake_series(0, "s0_bone", 30, 5.0, kernel="BONE", desc="CT BONE", start=3)
               + fake_series(0, "s0_5mm", 30, 5.0, desc="CT 5mm", start=4)
               + fake_series(0, "s0_3mm", 50, 3.0, desc="CT 3mm", start=5)
               + fake_series(1, "s1_thin", 200, 0.625, desc="CT PLAIN THIN", start=2)   # only thin -> fallback
               + fake_series(1, "s1_loc", 2, 5.0, image_type=("ORIGINAL", "PRIMARY", "LOCALIZER"), start=1)
               + fake_series(2, "s2_c", 30, 5.0, contrast="OMNIPAQUE", start=2)          # contrast only -> excluded
               + fake_series(3, "s3_5mm", 30, 5.0, start=2)
               + fake_series(3, "s3_5mm_b", 30, 5.0, start=3))                           # BHX on this one
    headers.append(None)                                                                # unreadable file
    inv, files = build_inventory(headers)
    assert len(inv) == 9 and len(files) == 30 * 5 + 200 * 2 + 50 + 2
    assert set(inv["case_id"]) == {"CQ500-CT-0", "CQ500-CT-1", "CQ500-CT-2", "CQ500-CT-3"}

    labels = pd.DataFrame({"case_id": ["CQ500-CT-0", "CQ500-CT-1", "CQ500-CT-2", "CQ500-CT-3", "CQ500-CT-4"],
                           "category": ["B1", "B1", "B2", "B2", "B2"], "ICH": [0, 0, 0, 1, 1],
                           "ICH_votes": [0, 0, 0, 3, 2], "ICH_unanimous": [1, 1, 1, 1, 0],
                           "abnormal_any_reader": [0, 0, 0, 1, 1]})
    bhx = pd.DataFrame({"series_uid": ["s3_5mm_b"], "bhx_boxes": [4], "bhx_images": [3]})
    sel, excl = select(inv, labels, bhx)
    pick = dict(zip(sel["case_id"], sel["series_uid"]))
    assert pick["CQ500-CT-0"] == "s0_5mm"                     # closest to 5 mm, not bone, not thin
    assert pick["CQ500-CT-1"] == "s1_thin"                    # thin-slice fallback ...
    assert sel.set_index("case_id").loc["CQ500-CT-1", "fallback_thin"]   # ... and flagged
    assert pick["CQ500-CT-3"] == "s3_5mm_b"                   # BHX tie-breaker wins over series number
    assert "CQ500-CT-2" not in pick and "CQ500-CT-4" not in pick
    reasons = dict(zip(excl["case_id"], excl["reason"]))
    assert "no eligible" in reasons["CQ500-CT-2"] and "no DICOM" in reasons["CQ500-CT-4"]


def test_bhx_box_parsing():
    assert parse_box("{'x': 263.9, 'y': 252.0, 'width': 26.7, 'height': 22.6}")["width"] == 26.7
    assert parse_box('{"x": 1, "y": 2, "width": 3, "height": 4}') == {"x": 1, "y": 2, "width": 3, "height": 4}


if __name__ == "__main__":
    test_case_ids()
    test_labels_majority_and_kappa()
    test_series_geometry()
    test_inventory_and_selection()
    test_bhx_box_parsing()
    print("all CQ500 data checks passed")
