import os
from pathlib import Path

# ----------------------------------------------------------------------
#  File locations
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_PMIDS_FILE = DATA_DIR / "processed_data.pkl"