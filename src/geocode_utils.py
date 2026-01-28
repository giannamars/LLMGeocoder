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
    """Return query strings ordered from most specific → least specific."""
    region = row.get("region", "").strip()
    country = row.get("country", "").strip()
    loc = row.get("location_name", "").strip()

    def _join(parts: List[str]) -> str:
        return ", ".join(p for p in parts if p)

    candidates: List[str] = []

    # Full string
    candidates.append(_join([region, country, loc]))

    # Progressively shorten location name
    if loc:
        tokens = loc.split()
        for n in range(len(tokens) - 1, 0, -1):
            shortened = " ".join(tokens[:n])
            candidates.append(_join([region, country, shortened]))

    # Region + country only
    candidates.append(_join([region, country]))

    # Country only
    candidates.append(country)

    # Deduplicate while preserving order
    seen = set()
    return [q for q in candidates if q and not (q in seen or seen.add(q))]


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