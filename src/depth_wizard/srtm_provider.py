"""Public elevation samples with explicit source and support provenance.

Copernicus GLO-30 is a digital *surface* model. Neither these samples nor
radar-derived SRTM elevations certify bare earth at a pixel. A 5x5 sample
interpolation does not create fine-resolution terrain observations. API
failure or NODATA raises an error; no invented topography is substituted.
"""

import numpy as np
from typing import Dict, Any
import json

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


class ElevationSourceError(RuntimeError):
    """A requested elevation source could not supply valid observations."""

    def __init__(self, message: str, provenance: Dict[str, Any]):
        super().__init__(message)
        self.provenance = provenance


class SRTMElevationProvider:
    """Sample public elevation APIs, never treating them as certified DTM."""

    # Cache to avoid redundant API calls for the same location
    _cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _validate_coordinates(lat: float, lon: float):
        if not (np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Elevation coordinates must be finite WGS84 latitude/longitude.")

    @staticmethod
    def _validate_size(value: int, name: str):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= value <= 2048:
            raise ValueError(f"{name} must be an integer between 1 and 2048.")

    @staticmethod
    def _valid_elevations(values, count: int) -> np.ndarray:
        result = np.ma.asarray(values, dtype=np.float64).filled(np.nan).reshape(-1)
        if result.size != count or not np.all(np.isfinite(result)) or np.any((result < -500) | (result > 9000)):
            raise ValueError("Elevation source returned missing, NODATA, or invalid samples.")
        return result

    @staticmethod
    def _provenance(source: str, count: int) -> Dict[str, Any]:
        copernicus = source.startswith("OPEN_METEO")
        return {
            "status": "AVAILABLE",
            "source": source,
            "dataset": "COPERNICUS_DEM_REPORTED_BY_SERVICE" if copernicus else "UNSPECIFIED_BY_OPEN_ELEVATION_RESPONSE",
            "dataset_verified": False,
            "product_resolution_m": None,
            "source_surface_type": "DSM" if copernicus else "ELEVATION_SURFACE_UNVERIFIED",
            "bare_earth_certified": False,
            "vertical_datum": "UNKNOWN",
            "nominal_dataset_vertical_datum": "EGM2008" if copernicus else "UNKNOWN",
            "datum_verified": False,
            "horizontal_crs": "EPSG:4326",
            "independent_of_image": True,
            "support": {"valid_sample_count": count, "requested_sample_count": count},
            "limitations": [
                "Elevation API does not verify the returned vertical datum.",
                "Product version and resolution are not identified in the elevation response.",
                "Surface elevations are not certified bare-earth or structural-height controls.",
            ],
        }

    @classmethod
    def _fetch_samples(cls, lats, lons, timeout_s: float):
        if not np.isfinite(timeout_s) or not 0 < timeout_s <= 30:
            raise ValueError("timeout_s must be positive and no greater than 30 seconds.")
        count = len(lats)
        attempts = []
        if HAS_URLLIB:
            urls = [
                ("OPEN_METEO_COPERNICUS_DEM", "https://api.open-meteo.com/v1/elevation?latitude="
                 + ",".join(f"{v:.8f}" for v in lats) + "&longitude="
                 + ",".join(f"{v:.8f}" for v in lons)),
                ("OPEN_ELEVATION", "https://api.open-elevation.com/api/v1/lookup?locations="
                 + "|".join(f"{lat:.8f},{lon:.8f}" for lat, lon in zip(lats, lons))),
            ]
            for source, url in urls:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "DepthWizard-SIH26175/2.0"})
                    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                        data = json.loads(resp.read().decode())
                    if source == "OPEN_METEO_COPERNICUS_DEM":
                        values = data.get("elevation", [])
                    else:
                        values = [item.get("elevation") for item in data.get("results", [])]
                    elevations = cls._valid_elevations(values, count)
                    provenance = cls._provenance(source, count)
                    provenance["failed_source_attempts"] = attempts
                    return elevations, provenance
                except Exception as exc:
                    attempts.append({"source": source, "status": "UNAVAILABLE_OR_INVALID", "error_type": type(exc).__name__})
        raise ElevationSourceError("No public elevation source supplied valid observations.", {
            "status": "UNAVAILABLE", "source": "NONE", "vertical_datum": "UNKNOWN",
            "bare_earth_certified": False,
            "support": {"valid_sample_count": 0, "requested_sample_count": count},
            "failed_source_attempts": attempts,
        })

    @classmethod
    def get_elevation_at_point(
        cls,
        lat: float,
        lon: float,
        timeout_s: float = 5.0,
        return_metadata: bool = False
    ):
        """
        Fetch an observed elevation. Metadata describes surface/datum limitations.
        The legacy float return is retained; failures raise ElevationSourceError.
        """
        cls._validate_coordinates(lat, lon)
        if not np.isfinite(timeout_s) or not 0 < timeout_s <= 30:
            raise ValueError("timeout_s must be positive and no greater than 30 seconds.")
        cache_key = f"{lat:.8f}_{lon:.8f}"
        if cache_key in cls._cache:
            record = cls._cache[cache_key]
        else:
            elevations, provenance = cls._fetch_samples([lat], [lon], timeout_s)
            record = {"elevation_m": float(elevations[0]), **provenance}
            cls._cache[cache_key] = record
        if return_metadata:
            from copy import deepcopy
            return deepcopy(record)
        return record["elevation_m"]

    @classmethod
    def get_elevation_grid(
        cls,
        bounds: list,
        grid_rows: int = 512,
        grid_cols: int = 512,
        timeout_s: float = 8.0,
        return_metadata: bool = False
    ):
        """
        Bilinearly interpolate 25 elevation samples (row zero is north).
        Metadata mode returns {grid, provenance, status, source, support, ...}.
        Without metadata, the legacy array return remains available. No centre
        point or analytical plane is substituted when the grid is unavailable.
        """
        cls._validate_size(grid_rows, "grid_rows")
        cls._validate_size(grid_cols, "grid_cols")
        if not isinstance(bounds, (list, tuple, np.ndarray)) or len(bounds) != 4:
            raise ValueError("bounds must be [west, south, east, north] in WGS84.")
        west, south, east, north = [float(v) for v in bounds]
        cls._validate_coordinates(south, west)
        cls._validate_coordinates(north, east)
        if west >= east or south >= north:
            raise ValueError("bounds must have positive width and height without dateline wrapping.")
        n_samples = 5
        sample_lats = np.linspace(north, south, n_samples)
        sample_lons = np.linspace(west, east, n_samples)
        all_lats = np.repeat(sample_lats, n_samples)
        all_lons = np.tile(sample_lons, n_samples)
        values, provenance = cls._fetch_samples(all_lats, all_lons, timeout_s)
        elevations = values.reshape(n_samples, n_samples)

        from scipy.ndimage import map_coordinates
        yy, xx = np.meshgrid(np.linspace(0, n_samples - 1, grid_rows),
                             np.linspace(0, n_samples - 1, grid_cols), indexing="ij")
        grid = map_coordinates(elevations, [yy, xx], order=1, mode="nearest").astype(np.float32)
        provenance["support"].update({
            "sample_grid_dimensions": [n_samples, n_samples],
            "output_grid_dimensions": [int(grid_rows), int(grid_cols)],
            "interpolation": "BILINEAR_FROM_25_SOURCE_SAMPLES",
            "fine_resolution_observation": False,
            "bounds_wgs84": [west, south, east, north],
            "row_order": "NORTH_TO_SOUTH",
        })
        if return_metadata:
            return {"grid": grid, "provenance": provenance, **provenance}
        return grid

    @classmethod
    def clear_cache(cls):
        """Clears the elevation cache."""
        cls._cache.clear()
