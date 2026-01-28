import os
from pathlib import Path

# ----------------------------------------------------------------------
#  File locations
# ----------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Auto-create if missing

PROCESSED_PMIDS_FILE = DATA_DIR / "processed_data.pkl"