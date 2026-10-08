"""Live satellite image acquisition and geocoding for DepthWizard.

Provides zero-credential satellite image downloading and place-name geocoding
so the system can autonomously process any location without manual uploads.

APIs used (all free, no key required for core functionality):
- Nominatim (OpenStreetMap): Geocoding
- ArcGIS World Imagery: Satellite tiles (primary -- unlimited, no key)
- Mapbox Static: High-res satellite (backup -- free tier 100K/month)
- Open-Meteo Elevation: Live DEM anchoring
"""

from __future__ import annotations

import io
import json
import math
import os
import time
import threading
import urllib.request
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

_cache_lock = threading.Lock()
_cache_store: Dict[str, Tuple[float, Any]] = {}


def _cached(key: str, ttl_seconds: float, fetcher):
    now = time.monotonic()
    with _cache_lock:
        if key in _cache_store and _cache_store[key][0] > now:
            return _cache_store[key][1]
    result = fetcher()
    with _cache_lock:
        _cache_store[key] = (now + ttl_seconds, result)
    return result


REQUEST_HEADERS = {"User-Agent": "DepthWizard-SIH26175/2.0 (live-imagery-adapter)"}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def geocode_place(query: str, limit: int = 5) -> Dict[str, Any]:
    """Geocode a place name to coordinates using OpenStreetMap Nominatim."""
    if not query or not query.strip():
        return {"status": "error", "message": "Empty query string."}

    def _fetch():
        params = urllib.parse.urlencode({
            "q": query.strip(),
            "format": "json",
            "limit": min(max(limit, 1), 10),
            "addressdetails": 1,
            "extratags": 1,
        })
        url = f"{NOMINATIM_URL}?{params}"
        req = urllib.request.Request(url, headers={
            **REQUEST_HEADERS, "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            return {"status": "unavailable", "message": f"Nominatim request failed: {type(exc).__name__}: {exc}"}

        if not data:
            return {"status": "not_found", "message": f"No results for '{query}'."}

        results = []
        for item in data:
            try:
                lat = float(item["lat"])
                lon = float(item["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            bbox = item.get("boundingbox", [])
            address = item.get("address", {})
            results.append({
                "lat": lat,
                "lon": lon,
                "display_name": item.get("display_name", query),
                "place_type": item.get("type", "unknown"),
                "category": item.get("category", "unknown"),
                "importance": item.get("importance"),
                "country": address.get("country", ""),
                "country_code": address.get("country_code", ""),
                "state": address.get("state", ""),
                "city": address.get("city") or address.get("town") or address.get("village", ""),
                "bounding_box": {
                    "min_lat": float(bbox[0]) if len(bbox) > 0 else lat - 0.05,
                    "max_lat": float(bbox[1]) if len(bbox) > 1 else lat + 0.05,
                    "min_lon": float(bbox[2]) if len(bbox) > 2 else lon - 0.05,
                    "max_lon": float(bbox[3]) if len(bbox) > 3 else lon + 0.05,
                },
            })
        return {
            "status": "available",
            "query": query,
            "results": results,
            "primary": results[0] if results else None,
            "count": len(results),
            "provider": "OpenStreetMap Nominatim",
            "provider_url": "https://nominatim.openstreetmap.org",
        }

    cache_key = f"geocode:{query.strip().lower()}"
    return _cached(cache_key, 3600, _fetch)


ARCGIS_WORLD_IMAGERY_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
)


def _tile_bbox(lat: float, lon: float, zoom: int, size_px: int = 512) -> Tuple[float, float, float, float]:
    """Compute a bounding box around a coordinate for a given zoom level."""
    meters_per_pixel = 156543.03 * math.cos(math.radians(lat)) / (2 ** zoom)
    half_extent_m = meters_per_pixel * size_px / 2.0
    dlat = half_extent_m / 111320.0
    dlon = half_extent_m / (111320.0 * max(math.cos(math.radians(lat)), 0.01))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def fetch_satellite_image_arcgis(
    lat: float, lon: float, zoom: int = 16, size_px: int = 512,
) -> Dict[str, Any]:
    """Download satellite imagery from ArcGIS World Imagery (free, no key, unlimited)."""
    base = {
        "provider": "Esri ArcGIS World Imagery (Maxar/Airbus/USDA)",
        "provider_url": "https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9",
        "coordinates": {"lat": lat, "lon": lon},
        "zoom": zoom,
        "size_px": size_px,
    }

    bbox = _tile_bbox(lat, lon, zoom, size_px)
    params = urllib.parse.urlencode({
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{size_px},{size_px}",
        "format": "png",
        "f": "image",
        "transparent": "false",
    })
    url = f"{ARCGIS_WORLD_IMAGERY_URL}?{params}"

    try:
        req = urllib.request.Request(url, headers=REQUEST_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            image_bytes = resp.read()
            content_type = resp.headers.get("Content-Type", "")
    except Exception as exc:
        return {**base, "status": "unavailable", "message": f"ArcGIS request failed: {type(exc).__name__}: {exc}"}

    if not image_bytes or len(image_bytes) < 1000:
        return {**base, "status": "unavailable", "message": "ArcGIS returned empty or invalid image."}

    if "json" in content_type.lower() or image_bytes[:1] == b"{":
        return {**base, "status": "unavailable", "message": "ArcGIS returned an error response."}

    image_array = None
    if HAS_CV2:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_array is not None:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    meters_per_pixel = 156543.03 * math.cos(math.radians(lat)) / (2 ** zoom)

    return {
        **base,
        "status": "available",
        "image_bytes": image_bytes,
        "image_array": image_array,
        "content_type": content_type or "image/png",
        "bytes_size": len(image_bytes),
        "ground_resolution_m": round(meters_per_pixel, 3),
        "bounding_box": {
            "min_lon": bbox[0], "min_lat": bbox[1],
            "max_lon": bbox[2], "max_lat": bbox[3],
        },
        "is_live": True,
    }


def fetch_satellite_image_mapbox(
    lat: float, lon: float, zoom: int = 16, size_px: int = 512,
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Download satellite imagery from Mapbox Static API (free tier: 100K/month)."""
    token = access_token or os.getenv("MAPBOX_ACCESS_TOKEN")
    base = {
        "provider": "Mapbox Satellite",
        "provider_url": "https://docs.mapbox.com/api/maps/static-images/",
        "coordinates": {"lat": lat, "lon": lon},
        "zoom": zoom,
        "size_px": size_px,
    }
    if not token:
        return {
            **base, "status": "not_configured",
            "message": "Set MAPBOX_ACCESS_TOKEN for Mapbox satellite imagery (free: 100K tiles/month at mapbox.com).",
        }
    size = min(max(size_px, 64), 1280)
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"{lon:.6f},{lat:.6f},{zoom},0/{size}x{size}@2x"
        f"?access_token={token}"
    )
    try:
        req = urllib.request.Request(url, headers=REQUEST_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            image_bytes = resp.read()
    except Exception as exc:
        return {**base, "status": "unavailable", "message": f"Mapbox request failed: {type(exc).__name__}: {exc}"}
    if not image_bytes or len(image_bytes) < 1000:
        return {**base, "status": "unavailable", "message": "Mapbox returned empty or invalid image."}
    image_array = None
    if HAS_CV2:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_array is not None:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    meters_per_pixel = 156543.03 * math.cos(math.radians(lat)) / (2 ** zoom)
    return {
        **base, "status": "available", "image_bytes": image_bytes,
        "image_array": image_array, "content_type": "image/png",
        "bytes_size": len(image_bytes),
        "ground_resolution_m": round(meters_per_pixel / 2, 3),
        "is_live": True,
    }


def fetch_best_satellite_image(
    lat: float, lon: float, zoom: int = 16, size_px: int = 512,
) -> Dict[str, Any]:
    """Try ArcGIS (free, no key) first, then Mapbox as fallback."""
    result = fetch_satellite_image_arcgis(lat, lon, zoom, size_px)
    if result.get("status") == "available":
        result["source_priority"] = "primary"
        return result
    mapbox_result = fetch_satellite_image_mapbox(lat, lon, zoom, size_px)
    if mapbox_result.get("status") == "available":
        mapbox_result["source_priority"] = "fallback"
        mapbox_result["primary_failure"] = result.get("message", "ArcGIS unavailable")
        return mapbox_result
    return {
        "status": "unavailable",
        "message": "All satellite imagery providers failed.",
        "attempts": [
            {"provider": "ArcGIS", "error": result.get("message")},
            {"provider": "Mapbox", "error": mapbox_result.get("message")},
        ],
        "coordinates": {"lat": lat, "lon": lon},
    }


def fetch_live_elevation_grid(
    lat: float, lon: float, grid_size: int = 9, spacing_m: float = 300.0,
) -> Dict[str, Any]:
    """Fetch a grid of elevation points from Open-Meteo Copernicus DEM."""
    grid_size = min(max(grid_size, 3), 9)
    half = (grid_size - 1) / 2.0
    dlat = spacing_m / 111320.0
    dlon = spacing_m / (111320.0 * max(math.cos(math.radians(lat)), 0.01))

    lats = [lat + (i - half) * dlat for i in range(grid_size)]
    lons = [lon + (j - half) * dlon for j in range(grid_size)]

    flat_lats = [la for la in lats for _ in lons]
    flat_lons = [lo for _ in lats for lo in lons]

    lat_str = ",".join(f"{v:.8f}" for v in flat_lats)
    lon_str = ",".join(f"{v:.8f}" for v in flat_lons)
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"

    try:
        req = urllib.request.Request(url, headers=REQUEST_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        return {
            "status": "unavailable",
            "message": f"Elevation API failed: {type(exc).__name__}: {exc}",
            "provider": "Open-Meteo Copernicus DEM",
        }

    elevations = data.get("elevation", [])
    if len(elevations) != grid_size * grid_size:
        return {
            "status": "unavailable",
            "message": f"Expected {grid_size*grid_size} elevation values, got {len(elevations)}.",
        }

    elev_grid = np.array(elevations, dtype=np.float64).reshape(grid_size, grid_size)
    valid = np.isfinite(elev_grid) & (elev_grid > -500) & (elev_grid < 9000)

    if not np.all(valid):
        return {
            "status": "partial",
            "message": f"Some elevation values are invalid ({np.sum(~valid)} of {grid_size*grid_size}).",
            "elevation_grid": elev_grid.tolist(),
            "valid_count": int(np.sum(valid)),
        }

    return {
        "status": "available",
        "provider": "Open-Meteo Copernicus DEM (GLO-30)",
        "center": {"lat": lat, "lon": lon},
        "grid_size": grid_size,
        "spacing_m": spacing_m,
        "elevation_grid": elev_grid.tolist(),
        "stats": {
            "min_m": float(np.min(elev_grid)),
            "max_m": float(np.max(elev_grid)),
            "mean_m": float(np.mean(elev_grid)),
            "range_m": float(np.max(elev_grid) - np.min(elev_grid)),
        },
        "base_elevation_m": float(np.median(elev_grid)),
        "max_structural_height_m": min(max(float(np.max(elev_grid) - np.min(elev_grid)), 30.0), 950.0),
        "is_live": True,
    }
