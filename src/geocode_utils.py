# geocode_utils.py
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.location import Location as GeoLocation
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------------------------------------------
# Global geolocator
# ----------------------------------------------------------------------
geolocator = Nominatim(
    user_agent="llm_sr_pathogen (your.email@domain.com)",
    timeout=10,
)

# ----------------------------------------------------------------------
# On-disk cache
# ----------------------------------------------------------------------
CACHE_PATH = Path("geocode_cache.json")
_CACHE: Dict[str, Tuple[Optional[float], Optional[float]]] = (
    json.loads(CACHE_PATH.read_text()) if CACHE_PATH.is_file() else {}
)


def _save_cache() -> None:
    """Write the in-memory cache to disk."""
    CACHE_PATH.write_text(json.dumps(_CACHE, ensure_ascii=False, indent=2))


# ----------------------------------------------------------------------
# Query candidate builder
# ----------------------------------------------------------------------
def build_geocode_candidates(row: Dict[str, Any]) -> List[str]:
    """
    Return query strings ordered from most specific → least specific.
    Prioritizes location_name (e.g., hospital name) for better matching.
    """
    # Extract all location fields
    region = row.get("region", "").strip()
    country = row.get("country", "").strip()
    location_name = row.get("location_name", "").strip()
    amenity = row.get("amenity", "").strip()
    street = row.get("street", "").strip()
    city = row.get("city", "").strip()
    county = row.get("county", "").strip()
    state = row.get("state", "").strip()
    postalcode = row.get("postalcode", "").strip()

    # Filter out "unknown" values
    def _clean(val: str) -> str:
        return "" if val.lower() == "unknown" else val

    region = _clean(region)
    country = _clean(country)
    location_name = _clean(location_name)
    amenity = _clean(amenity)
    street = _clean(street)
    city = _clean(city)
    county = _clean(county)
    state = _clean(state)
    postalcode = _clean(postalcode)

    def _join(parts: List[str]) -> str:
        return ", ".join(p for p in parts if p)

    candidates: List[str] = []

    # 1. Location name + city + state + country (most specific for hospitals)
    #    e.g., "Bintulu Hospital, Bintulu, Sarawak, Malaysia"
    candidates.append(_join([location_name, city, state, country]))

    # 2. Location name + city + country
    #    e.g., "Bintulu Hospital, Bintulu, Malaysia"
    candidates.append(_join([location_name, city, country]))

    # 3. Location name + state + country
    #    e.g., "Bintulu Hospital, Sarawak, Malaysia"
    candidates.append(_join([location_name, state, country]))

    # 4. Location name + country only
    #    e.g., "Bintulu Hospital, Malaysia"
    candidates.append(_join([location_name, country]))

    # 5. City + state + country (fallback to city level)
    #    e.g., "Bintulu, Sarawak, Malaysia"
    candidates.append(_join([city, state, country]))

    # 6. City + country
    #    e.g., "Bintulu, Malaysia"
    candidates.append(_join([city, country]))

    # 7. Street address if available
    if street:
        candidates.append(_join([street, city, state, country]))

    # 8. Postalcode + country
    if postalcode:
        candidates.append(_join([postalcode, country]))

    # 9. State + country
    candidates.append(_join([state, country]))

    # 10. Country only (last resort)
    candidates.append(country)

    # Deduplicate while preserving order
    seen = set()
    return [q for q in candidates if q and not (q in seen or seen.add(q))]

async def _geocode_with_fallback(
    row: Dict[str, Any]
) -> Tuple[Optional[float], Optional[float], str]:
    """Try most-specific query first, then progressively less-specific."""
    candidates = build_geocode_candidates(row)
    
    logging.debug(f"Geocode candidates for {row.get('location_name')}: {candidates[:5]}")

    for query in candidates:
        lat, lon, success = await _geocode_async(query)
        logging.debug(f"  Tried '{query}' → {'✓' if success else '✗'}")
        if success:
            return lat, lon, query

    return None, None, "not_found"

# ----------------------------------------------------------------------
# Core blocking geocode function
# ----------------------------------------------------------------------
def _geocode_one(query: str) -> Tuple[Optional[float], Optional[float], bool]:
    """
    Returns (lat, lon, success). success is True if found, False otherwise.
    """
    if query in _CACHE:
        lat, lon = _CACHE[query]
        return lat, lon, lat is not None

    try:
        location: Optional[GeoLocation] = geolocator.geocode(query)
        if location is None:
            _CACHE[query] = (None, None)
            return None, None, False

        lat, lon = location.latitude, location.longitude
        _CACHE[query] = (lat, lon)
        return lat, lon, True

    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        logging.warning(f"Nominatim error for '{query}': {exc}")
        return None, None, False


# ----------------------------------------------------------------------
# Async wrapper
# ----------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=4)


async def _geocode_async(query: str) -> Tuple[Optional[float], Optional[float], bool]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _geocode_one, query)


# ----------------------------------------------------------------------
# Geocode with fallback (tries progressively less specific queries)
# ----------------------------------------------------------------------
async def _geocode_with_fallback(
    row: Dict[str, Any]
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Try most-specific query first, then progressively less-specific.
    Returns (lat, lon, matched_query) where matched_query is the successful
    query string or "not_found".
    """
    candidates = build_geocode_candidates(row)

    for query in candidates:
        lat, lon, success = await _geocode_async(query)
        if success:
            return lat, lon, query  # Return the successful query string

    return None, None, "not_found"


# ----------------------------------------------------------------------
# Helper to check if row should skip geocoding
# ----------------------------------------------------------------------
def _is_all_unknown(row: Dict[str, Any]) -> bool:
    """Return True if region, country, and location_name are all 'unknown'."""
    for key in ("country", "location_name", "region"):
        val = row.get(key)
        if not isinstance(val, str) or val.strip().lower() != "unknown":
            return False
    return True


# ----------------------------------------------------------------------
# Public: add geocode columns to rows
# ----------------------------------------------------------------------
async def add_geocode_columns(
    rows: List[Dict[str, Any]],
    *,
    pause: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Enrich each row with latitude, longitude, and geocode_query.
    geocode_query contains the successful query string or "not_found"/"skipped".
    """
    enriched: List[Dict[str, Any]] = []

    for row in rows:
        # Fast-path: all fields are "unknown"
        if _is_all_unknown(row):
            row.update(latitude=None, longitude=None, geocode_query="not_found")
            enriched.append(row)
            continue

        # Need at least country and location_name
        country = row.get("country")
        location_name = row.get("location_name")

        if not country or not location_name:
            row.update(latitude=None, longitude=None, geocode_query="not_found")
            enriched.append(row)
            continue

        # Geocode with fallback
        lat, lon, matched_query = await _geocode_with_fallback(row)

        if pause:
            await asyncio.sleep(pause)

        row.update(latitude=lat, longitude=lon, geocode_query=matched_query)
        enriched.append(row)

    _save_cache()
    return enriched


# ----------------------------------------------------------------------
# No-op geocode (when GEOCODE=False)
# ----------------------------------------------------------------------
async def noop_geocode(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identity coroutine - returns rows with placeholder geocode columns."""
    for row in rows:
        row.setdefault("latitude", None)
        row.setdefault("longitude", None)
        row.setdefault("geocode_query", "skipped")
    return rows