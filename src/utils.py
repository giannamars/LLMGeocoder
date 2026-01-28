# src/utils.py
import re
import time
import logging
import pickle
from typing import List, Optional, Dict, Any, Mapping

from geopy.geocoders import Nominatim
import spacy

from src.config import PROCESSED_PMIDS_FILE


# ----------------------------------------------------------------------
# Accession number extraction patterns (categorized)
# ----------------------------------------------------------------------
_ACCESSION_CATEGORIES = {
    "bioproject": re.compile(r"\bPRJ[A-Z]{2}[0-9]+\b"),
    "biosample": re.compile(r"\bSAM[A-Z]{1,2}[0-9]+\b"),
    "sra": re.compile(r"\b[SED]R[RSXP][0-9]{6,9}\b"),
    "genbank": re.compile(r"\b[A-Z]{1,2}[0-9]{5,8}\b"),
    "refseq": re.compile(r"\b(?:NC|NZ|NW)_[A-Z0-9\.]+\b"),
    "assembly": re.compile(r"\bGC[AF]_[0-9]{9}(?:\.[0-9]+)?\b"),
    "ena": re.compile(r"\bER[RSXP][0-9]{6,9}\b"),
    "ddbj": re.compile(r"\bDR[RSXP][0-9]{6,9}\b"),
    "uniprot": re.compile(r"\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b"),
    "gisaid": re.compile(r"\bEPI_ISL_[0-9]+\b"),
    "wgs": re.compile(r"\b[A-Z]{4}[0-9]{8,10}\b"),
}


def _extract_accessions_categorized(text: str) -> Dict[str, List[str]]:
    """
    Extract accessions and categorize by database.
    Returns dict with database names as keys and lists of accessions as values.
    """
    if not isinstance(text, str):
        return {}
    
    result = {}
    for name, pattern in _ACCESSION_CATEGORIES.items():
        matches = pattern.findall(text)
        # Deduplicate while preserving order
        seen = set()
        unique = [m for m in matches if not (m in seen or seen.add(m))]
        # Filter short matches (likely false positives)
        unique = [m for m in unique if len(m) >= 6]
        if unique:
            result[name] = unique
    
    return result


def _extract_accessions_from_text(text: str) -> List[str]:
    """
    Scan text for genome-related accession identifiers.
    Returns deduplicated flat list preserving order.
    """
    categorized = _extract_accessions_categorized(text)
    # Flatten all categories into a single list
    all_accessions = []
    seen = set()
    for accessions in categorized.values():
        for acc in accessions:
            if acc not in seen:
                seen.add(acc)
                all_accessions.append(acc)
    return all_accessions


def extract_genome_accession_numbers(text: str) -> Optional[str]:
    """
    Scan text for accession identifiers, return as semicolon-separated string.
    Returns None if no matches found.
    """
    accessions = _extract_accessions_from_text(text)
    return "; ".join(accessions) if accessions else None


# ----------------------------------------------------------------------
# spaCy model (lazy-loaded, cached)
# ----------------------------------------------------------------------
_nlp: Optional["spacy.language.Language"] = None


def _get_nlp() -> "spacy.language.Language":
    """Return a cached spaCy English pipeline."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Install it with: python -m spacy download en_core_web_sm"
            ) from exc
    return _nlp


# ----------------------------------------------------------------------
# Hospital extraction
# ----------------------------------------------------------------------
def extract_hospital_info(
    text: str,
    doc_metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[List[str]]:
    """
    Extract hospital information using NER, regex fallback, then metadata fallback.
    Returns list of hospital names or None if no hospital mentioned.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    # NER-based extraction
    hospital_names = [
        ent.text
        for ent in doc.ents
        if ent.label_ in {"ORG", "FACILITY"} and "hospital" in ent.text.lower()
    ]
    if hospital_names:
        return hospital_names

    # Regex-based extraction
    hospital_mentions = re.findall(r"(hospital|our hospital)", text, flags=re.IGNORECASE)
    hospital_name_pattern = r"(?:hospital|our hospital)\s+([A-Za-z][A-Za-z0-9\s-]*)"
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    raw_matches = re.findall(hospital_name_pattern, text, flags=re.IGNORECASE)

    filtered_hospital_names: List[str] = []
    for match in raw_matches:
        lowered_match = match.lower()

        loc_found: Optional[str] = None
        for loc in locations:
            if loc.lower() in lowered_match:
                loc_found = loc
                break

        if loc_found:
            if lowered_match.rstrip().endswith(loc_found.lower()):
                trimmed = match[: -len(loc_found)].strip()
            else:
                trimmed = match.strip()

            tokens = trimmed.split()
            if tokens and tokens[-1].lower() in {"in", "at", "of"}:
                trimmed = " ".join(tokens[:-1])

            if trimmed:
                filtered_hospital_names.append(trimmed)

    if filtered_hospital_names:
        return filtered_hospital_names

    # Metadata fallback
    if hospital_mentions:
        author_affiliation = doc_metadata.get("affiliation") if doc_metadata else None
        return [author_affiliation] if author_affiliation else None

    return None


# ----------------------------------------------------------------------
# Coordinate extraction
# ----------------------------------------------------------------------
def extract_coordinates(text: str) -> Optional[str]:
    """Parse latitude/longitude pairs, return semicolon-separated string or None."""
    # Normalise typographic symbols
    replacements = {
        "o": "°", "'": "'", """: "\"", """: "\"",
        "'": "'", "′": "'", "″": "\"",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    pattern = r"""
        (\d{1,2}°\d{1,2}'\d{1,2}(?:\.\d+)?\"\s*[NS])
        \s*[;,]?\s*
        (\d{1,3}°\d{1,2}'\d{1,2}(?:\.\d+)?\"\s*[EW])
    """
    matches = re.findall(pattern, text, flags=re.VERBOSE)
    if not matches:
        return None
    return "; ".join(" ".join(pair) for pair in matches)


# ----------------------------------------------------------------------
# Geocoding (Nominatim)
# ----------------------------------------------------------------------
_geolocator = Nominatim(user_agent="burk-pseudomallei-geocoder")


def _geocode_one(place: str) -> Optional[Dict[str, Any]]:
    """Return {place, lat, lon} or None on failure."""
    if not place or place.strip().lower() == "unknown":
        return None

    try:
        location = _geolocator.geocode(place, exactly_one=True, timeout=10)
        if location:
            return {"place": place, "lat": location.latitude, "lon": location.longitude}
    except Exception as exc:
        logging.warning(f"Geocode error for '{place}': {exc}")
    return None


def geocode_hierarchy(hierarchy: str) -> Optional[Dict[str, Any]]:
    """Geocode from most specific to least specific component."""
    parts = [p.strip() for p in hierarchy.split("/") if p.strip().lower() != "unknown"]
    for part in reversed(parts):
        result = _geocode_one(part)
        if result:
            return result
        time.sleep(1)
    return None


def geocode_hospital(hospital_list: List[str]) -> List[Dict[str, Any]]:
    """Geocode a list of hospital names."""
    results = []
    for hosp in hospital_list:
        cleaned = hosp.strip().strip("[]")
        if cleaned.lower() == "unknown" or not cleaned:
            continue
        info = _geocode_one(cleaned)
        if info:
            results.append(info)
        time.sleep(1)
    return results


# ----------------------------------------------------------------------
# Metadata sanitization
# ----------------------------------------------------------------------
def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert non-primitive values to strings for Chroma compatibility."""
    clean: Dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


# ----------------------------------------------------------------------
# Location explosion
# ----------------------------------------------------------------------
def explode_locations(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten location list into one row per location."""
    locations = record.get("location")
    
    if not isinstance(locations, list) or not locations:
        return []

    # Fields to copy to each exploded row
    common_fields = [
        "pmid", "title", "study_type", "sample_date", "source",
        "retrieved_preview", "accession_numbers",
        "acc_bioproject", "acc_biosample", "acc_sra", "acc_genbank",
        "acc_refseq", "acc_assembly", "acc_ena", "acc_ddbj",
        "acc_uniprot", "acc_gisaid", "acc_wgs",
    ]

    exploded: List[Dict[str, Any]] = []

    for loc in locations:
        new_rec = {field: record.get(field) for field in common_fields}
        
        # Location-specific fields
        new_rec["region"] = loc.get("region", "unknown")
        new_rec["country"] = loc.get("country", "unknown")
        new_rec["location_name"] = loc.get("location", "unknown")
        new_rec["amenity"] = loc.get("amenity", "unknown")
        new_rec["street"] = loc.get("street", "unknown")
        new_rec["city"] = loc.get("city", "unknown")
        new_rec["county"] = loc.get("county", "unknown")
        new_rec["state"] = loc.get("state", "unknown")
        new_rec["postalcode"] = loc.get("postalcode", "unknown")
        new_rec["location"] = record.get("location")  # Keep original list
        
        exploded.append(new_rec)

    return exploded


# ----------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------
def load_processed_data() -> dict:
    """Read pickle storing processed PMIDs and results."""
    if PROCESSED_PMIDS_FILE.is_file():
        try:
            with PROCESSED_PMIDS_FILE.open("rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "pmids" in data:
                return data
        except Exception as exc:
            logging.warning(f"Could not read pickle: {exc}")
    return {"pmids": set(), "results": []}


def save_processed_data(data: dict) -> None:
    """Write pickle to disk."""
    with PROCESSED_PMIDS_FILE.open("wb") as f:
        pickle.dump(data, f)


# ----------------------------------------------------------------------
# Deduplication
# ----------------------------------------------------------------------
MAX_ROWS_PER_PMID: Optional[int] = None


def dedupe_and_limit_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate locations per PMID and optionally limit row count."""
    #logging.info(f"Dedupe input: {len(rows)} rows")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        pmid = r["pmid"]
        grouped.setdefault(pmid, []).append(r)

    cleaned: List[Dict[str, Any]] = []

    for pmid, pmid_rows in grouped.items():
        seen: set = set()
        uniq: List[Dict[str, Any]] = []

        for row in pmid_rows:
            # Use location_name (the exploded field), NOT location (the original list)
            key = (
                str(row.get("region", "unknown")),
                str(row.get("country", "unknown")),
                str(row.get("location_name", "unknown")),  # ← FIXED: use location_name
            )
            if key not in seen:
                seen.add(key)
                uniq.append(row)

        if MAX_ROWS_PER_PMID is not None and len(uniq) > MAX_ROWS_PER_PMID:
            uniq = uniq[:MAX_ROWS_PER_PMID]

        cleaned.extend(uniq)
    #logging.info(f"Dedupe output: {len(cleaned)} rows")

    return cleaned