# pickle_to_csv.py
"""Load processed_pmids.pkl and export to CSV."""

from pathlib import Path
import pickle
import pandas as pd
from typing import List, Dict, Any

from src.config import PROCESSED_PMIDS_FILE

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
    "latitude",
    "longitude",
    "geocode_query",
    # Accession columns
    "accession_numbers",
    "acc_bioproject",
    "acc_biosample",
    "acc_sra",
    "acc_genbank",
    "acc_refseq",
    "acc_assembly",
    "acc_ena",
    "acc_ddbj",
    "acc_uniprot",
    "acc_gisaid",
    "acc_wgs",
    "retrieved_preview",
]


def load_results_as_dataframe(
    pickle_path: Path = PROCESSED_PMIDS_FILE,
) -> pd.DataFrame:
    """Load pickle and return a DataFrame."""
    
    if not pickle_path.is_file():
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    with pickle_path.open("rb") as f:
        data = pickle.load(f)

    raw_rows = data.get("results", [])
    if not raw_rows:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    rows = [
        entry if isinstance(entry, dict) else entry.__dict__
        for entry in raw_rows
    ]

    df = pd.DataFrame(rows)

    # Convert list columns to semicolon-separated strings for CSV
    list_columns = [
        "accession_numbers", "acc_bioproject", "acc_biosample", "acc_sra",
        "acc_genbank", "acc_refseq", "acc_assembly", "acc_ena", "acc_ddbj",
        "acc_uniprot", "acc_gisaid", "acc_wgs",
    ]
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: "; ".join(x) if isinstance(x, list) else (x or "")
            )

    # Reorder columns
    existing_canonical = [c for c in CANONICAL_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    df = df[existing_canonical + extra_cols]

    # Add missing columns
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
    if drop_preview and "retrieved_preview" in df.columns:
        df = df.drop(columns=["retrieved_preview"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path.resolve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert pickle to CSV")
    parser.add_argument("-o", "--output", type=Path, default=Path("processed_pmids.csv"))
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--head", type=int, default=5)
    args = parser.parse_args()

    df = load_results_as_dataframe()
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    if args.head > 0 and len(df) > 0:
        print(f"\nFirst {args.head} rows:")
        # Show key columns including geocode
        cols_to_show = ["pmid", "location_name", "city", "latitude", "longitude", "geocode_query"]
        cols_available = [c for c in cols_to_show if c in df.columns]
        print(df[cols_available].head(args.head).to_string())

    csv_path = save_to_csv(df, args.output, drop_preview=args.no_preview)
    print(f"\n✅ CSV written to {csv_path}")