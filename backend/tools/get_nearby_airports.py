from __future__ import annotations
import os, json, math, heapq
from typing import Any, Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolve project root (repo root) regardless of current working directory
BASE_DIR = Path(__file__).resolve().parents[2]

# ---- Public tool spec (Responses API function tool) ----
TOOL_SPEC: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_nearby_airports",
        "description": "Find nearest airports to a place or coordinate. Returns up to max_results sorted by distance.",
        "parameters": {
            "type": "object",
            "properties": {
                "place": {"type": "string", "description": "Free-text place (e.g., 'Mount Whitney')."},
                "lat": {"type": "number", "description": "Latitude if already known."},
                "lng": {"type": "number", "description": "Longitude if already known."},
                "max_results": {"type": "integer", "default": 5},
                "max_radius_km": {"type": "number", "default": 250},
                "commercial_only": {"type": "boolean", "default": True},
                "airport_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": [
                        "large_airport","medium_airport","small_airport","heliport","seaplane_base"
                    ]},
                    "default": ["large_airport","medium_airport","small_airport"]
                }
            },
            "required": []
        }
    }
}

# Registry contract
TOOL_NAME = "get_nearby_airports"

def tool_definition() -> Dict[str, Any]:
    """Return the Responses API tool definition (wrapped with type/function)."""
    return TOOL_SPEC

# ---- Lightweight dataset loader ----
DEFAULT_AIRPORTS_JSON = os.getenv(
    "AIRPORTS_JSON_PATH",
    str(BASE_DIR / "data" / "airports_min.json"),
)
DEFAULT_AIRPORTS_CSV = os.getenv(
    "AIRPORTS_CSV_PATH",
    str(BASE_DIR / "data" / "ourairports" / "airports.csv"),
)
DEFAULT_PINS_JSON = os.getenv(
    "PINS_JSON_PATH",
    str(BASE_DIR / "data" / "pins.json"),
)

logger.info(f"Loading airports from {DEFAULT_AIRPORTS_JSON}")
logger.info(f"Loading pins from {DEFAULT_PINS_JSON}")
_AIRPORTS: Optional[List[Dict[str, Any]]] = None
_PINS: Dict[str, Dict[str, float]] = {}

def _load_pins() -> None:
    global _PINS
    try:
        with open(DEFAULT_PINS_JSON, "r", encoding="utf-8") as f:
            _PINS = json.load(f)
    except Exception:
        # Safe defaults; include Mount Whitney so OSS works out of the box
        _PINS = {
            "Mount Whitney": {"lat": 36.5786, "lng": -118.2923}
        }

def _load_airports() -> None:
    """Load minimal fields: name, ident, iata_code, type, lat, lng, scheduled_service, iso_region, iso_country, municipality, elevation_ft."""
    global _AIRPORTS
    if _AIRPORTS is not None:
        return
    _AIRPORTS = []
    # Prefer JSON
    if os.path.exists(DEFAULT_AIRPORTS_JSON):
        try:
            with open(DEFAULT_AIRPORTS_JSON, "r", encoding="utf-8") as f:
                rows = json.load(f)
            for r in rows:
                if "lat" in r and "lng" in r:
                    _AIRPORTS.append(r)
            return
        except Exception:
            _AIRPORTS = []

    # Fallback to CSV (OurAirports)
    if os.path.exists(DEFAULT_AIRPORTS_CSV):
        try:
            import csv
            with open(DEFAULT_AIRPORTS_CSV, newline="", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    if not r.get("latitude_deg") or not r.get("longitude_deg"):
                        continue
                    try:
                        lat = float(r["latitude_deg"])
                        lng = float(r["longitude_deg"])
                    except Exception:
                        continue
                    _AIRPORTS.append({
                        "name": r.get("name") or "",
                        "ident": r.get("ident") or "",
                        "iata_code": r.get("iata_code") or None,
                        "type": r.get("type") or "",
                        "lat": lat,
                        "lng": lng,
                        "scheduled_service": (r.get("scheduled_service","no") == "yes"),
                        "iso_region": r.get("iso_region") or "",
                        "iso_country": r.get("iso_country") or "",
                        "municipality": r.get("municipality") or "",
                        "elevation_ft": int(r["elevation_ft"]) if r.get("elevation_ft") else None,
                    })
        except Exception:
            _AIRPORTS = []
    # If neither file exists, keep empty → executor will return a helpful note.

# ---- Haversine + top-N ----
R_EARTH_KM = 6371.0088
def _haversine_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    lat1, lng1 = a; lat2, lng2 = b
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dl = math.radians(lng2 - lng1)
    s = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R_EARTH_KM * math.asin(min(1.0, math.sqrt(s)))

def _nearest(lat: float, lng: float, airports: List[Dict[str,Any]], k: int) -> List[Tuple[float, Dict[str,Any]]]:
    # Use a heap for nsmallest without pulling entire dataset into memory at once.
    return heapq.nsmallest(k, (
        (_haversine_km((lat,lng), (r["lat"], r["lng"])), r)
        for r in airports
    ), key=lambda x: x[0])

# ---- Optional geocoder (Nominatim), disabled by default ----
def _geocode(place: str) -> Optional[Tuple[float,float]]:
    provider = os.getenv("GEOCODER_PROVIDER", "none").lower()
    if not place:
        return None
    # Offline pins first
    if not _PINS:
        _load_pins()
    if place in _PINS:
        p = _PINS[place]
        return (float(p["lat"]), float(p["lng"]))
    if provider != "nominatim":
        return None  # stay offline by default

    import requests
    base = os.getenv("GEOCODER_BASE_URL", "https://nominatim.openstreetmap.org/search")
    try:
        r = requests.get(
            base,
            params={"q": place, "format": "jsonv2", "limit": 1, "addressdetails": 0},
            headers={"User-Agent": os.getenv("OSM_UA", "chat-with-rag/1.0 (contact@example.com)")},
            timeout=float(os.getenv("GEOCODER_TIMEOUT_SEC", "8"))
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return (float(data[0]["lat"]), float(data[0]["lon"]))
    except Exception:
        return None

# ---- Public executor (called by chat pipeline) ----
def execute(args: Dict[str, Any], chat_context: List[Dict[str,str]], *, existing_context: Optional[List[Dict[str,Any]]] = None) -> str:
    """
    Args come from the model tool-call. Return a concise text block for synthesis.
    Never raise: return a helpful message if data isn't configured.
    """
    _load_airports()
    if _AIRPORTS is None or len(_AIRPORTS) == 0:
        return ("Nearest Airports: dataset not configured.\n"
                "Add data/airports_min.json or data/ourairports/airports.csv to enable this tool.")

    place = (args or {}).get("place") or ""
    lat = (args or {}).get("lat")
    lng = (args or {}).get("lng")
    max_results = int((args or {}).get("max_results") or 5)
    max_radius_km = float((args or {}).get("max_radius_km") or 250.0)
    commercial_only = bool((args or {}).get("commercial_only", True))
    airport_types = (args or {}).get("airport_types") or ["large_airport","medium_airport","small_airport"]

    if (lat is None or lng is None):
        # Try to infer coordinates
        coord = _geocode(place)
        if not coord:
            # Last resort: respond gracefully
            return (f"Nearest Airports: couldn’t resolve coordinates for '{place or 'the place'}'. "
                    "Add it to data/pins.json or enable Nominatim (set GEOCODER_PROVIDER=nominatim).")
        lat, lng = coord

    # Filter airports locally
    pool = []
    for r in _AIRPORTS:
        if r.get("type") not in airport_types:
            continue
        if commercial_only and not r.get("scheduled_service", False):
            continue
        pool.append(r)

    if not pool:
        return "Nearest Airports: no airports match the current filters."

    # distance + radius
    nearest = _nearest(lat, lng, pool, k=max_results * 3)
    rows = []
    for dist_km, r in nearest:
        if dist_km > max_radius_km:
            continue
        rows.append({
            "name": r.get("name",""),
            "iata": r.get("iata_code") or "",
            "icao": r.get("ident") or "",
            "type": r.get("type") or "",
            "scheduled": bool(r.get("scheduled_service", False)),
            "lat": r.get("lat"),
            "lng": r.get("lng"),
            "elevation_ft": r.get("elevation_ft"),
            "city": r.get("municipality") or "",
            "region": (r.get("iso_region") or "").split("-")[-1],
            "country": r.get("iso_country") or "",
            "distance_km": round(dist_km, 1),
            "distance_mi": round(dist_km * 0.621371, 1),
        })
        if len(rows) >= max_results:
            break

    if not rows:
        return "Nearest Airports: none found within the specified radius."

    # Return a compact, synthesis-friendly text block
    lines = [f"Nearest airports to ({lat:.5f}, {lng:.5f}) — top {len(rows)}:"]
    for r in rows:
        codes = " • ".join(filter(None, [r["iata"], r["icao"]]))
        badge = " (commercial)" if r["scheduled"] else ""
        lines.append(f"- {r['name']} ({codes}) — {r['distance_mi']} mi / {r['distance_km']} km — {r['city']}, {r['region']}{badge}")
    # Attribution if we used Nominatim
    if os.getenv("GEOCODER_PROVIDER","none").lower() == "nominatim":
        lines.append("\nGeocoding by Nominatim • © OpenStreetMap contributors")
    return "\n".join(lines)

def run(args: Dict[str, Any], chat_context: List[Dict[str, str]], **kwargs) -> str:
    """Registry executor: delegates to execute()."""
    return execute(args or {}, chat_context or [], existing_context=kwargs.get("existing_context"))

__all__ = ["TOOL_NAME", "TOOL_SPEC", "tool_definition", "run", "execute"]
