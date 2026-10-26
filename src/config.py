"""Central path configuration.

Override the raw-data location without editing code:
    export PANTS_DATA_ROOT="/path/to/ct-fm-classifyer/data"
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

DATA_ROOT = Path(os.environ.get(
    "PANTS_DATA_ROOT",
    "/Users/bingerliang/Library/Mobile Documents/com~apple~CloudDocs/ct-fm-classifyer/data",
))
CT_DIR = DATA_ROOT / "PanTSMini_ImageTr_00000001_00001000"
LABEL_DIR = DATA_ROOT / "PanTSMini_Label 2"
CACHE_DIR = SPLIT_DIR / "cache_roi_pt"
METADATA_PATH = SPLIT_DIR / "metadata.xlsx"

# Head CT (CQ500 + BHX): override the location with  export CQ500_ROOT=/path/to/cq500_data
CQ500_ROOT = Path(os.environ.get("CQ500_ROOT", PROJECT_ROOT.parent / "cq500_data"))
CQ500_DICOM_DIR = CQ500_ROOT / "dicom"
CQ500_READS = CQ500_ROOT / "reads.csv"
BHX_DIR = CQ500_ROOT / "bhx"
BHX_MANUAL_CSV = BHX_DIR / "1_Initial_Manual_Labeling.csv"
HEADCT_DIR = SPLIT_DIR / "headct"
