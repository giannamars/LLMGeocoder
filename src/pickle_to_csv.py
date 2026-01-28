# pickle_to_csv.py
"""Load processed_pmids.pkl and export to CSV."""

from pathlib import Path
import pickle
import pandas as pd
from typing import List, Dict, Any

from src.config import PROCESSED_PMIDS_FILE
from src.utils import explode_locations

# ----------------------------------------------------------------------
# Canonical column order
# ----------------------------------------------------------------------
CANONICAL_COLUMNS = [
    "pmid",
    "title",
    "study_type",
    "sample_date",
    "region",
    "country",
    "location_name",
    "amenity",
    "street",
    "city",
    "county",
    "state",
    "postalcode",
    "accession_numbers",
    "retrieved_preview",
]


def load_results_as_dataframe(
    pickle_path: Path = PROCESSED_PMIDS_FILE,
) -> pd.DataFrame:
    """Load pickle and return a tidy DataFrame with exploded locations."""
    
    # Guard: return empty DF if file doesn't exist
    if not pickle_path.is_file():
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    # Load pickle
    with pickle_path.open("rb") as f:
        data = pickle.load(f)

    raw_rows = data.get("results", [])
    if not raw_rows:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    # Normalize entries to dicts
    rows = [
        entry if isinstance(entry, dict) else entry.__dict__
        for entry in raw_rows
    ]

    # Flatten locations (use generator to avoid intermediate list)
    flat_rows = [
        exploded_row
        for row in rows
        for exploded_row in (explode_locations(row) or [row])
    ]

    # Build DataFrame
    df = pd.DataFrame(flat_rows)

    # Reorder columns: canonical first, then extras
    existing_canonical = [c for c in CANONICAL_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    df = df[existing_canonical + extra_cols]

    # Add any missing canonical columns
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def save_to_csv(
    df: pd.DataFrame,
    output_path: Path = Path("processed_pmids.csv"),
    *,
    drop_preview: bool = False,
) -> Path:
    """
    Save DataFrame to CSV with optional optimizations.
    
    Parameters
    ----------
    df : DataFrame
    output_path : Path
    drop_preview : bool
        If True, exclude the 'retrieved_preview' column (saves space).
    
    Returns
    -------
    Path to the written CSV file.
    """
    if drop_preview and "retrieved_preview" in df.columns:
        df = df.drop(columns=["retrieved_preview"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path.resolve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert pickle to CSV")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("processed_pmids.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Exclude retrieved_preview column",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of rows to preview (0 to skip)",
    )
    args = parser.parse_args()

    df = load_results_as_dataframe()
    
    print(f"Loaded {len(df)} rows")
    
    if args.head > 0:
        print(f"\nFirst {args.head} rows:")
        print(df.head(args.head).to_string())

    csv_path = save_to_csv(df, args.output, drop_preview=args.no_preview)
    print(f"\n✅ CSV written to {csv_path}")