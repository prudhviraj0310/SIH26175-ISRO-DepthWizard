"""
DepthWizard LoD-1 Architectural Building Extrusion Engine
=========================================================
Extracts single-block Level-of-Detail 1 (LoD-1) architectural building prisms
from normalized Digital Surface Models (nDSM = DSM - DTM).

Features:
  1. Distance-transform marker-controlled watershed instance separation (with
     connected components fallback).
  2. Ramer-Douglas-Peucker (RDP) polygonal footprint simplification.
  3. Ground-anchored vertical extrusion for 3D web visualizers.

SCIENTIFIC & OPERATIONAL NOTE:
This module is an experimental visualization feature. Extrusion geometries
represent heuristic building block abstractions for interactive 3D rendering
and do not constitute certified cadastral building boundaries or surveyed heights.

References:
  - Biljecki, F. et al. (2016). "Formalisation of the Level of Detail concept for 3D city models."
    Computers, Environment and Urban Systems, 48, 1-15.
  - Ramer, U. (1972) / Douglas, D. & Peucker, T. (1973).
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def rdp_simplify(points: List[Tuple[float, float]], epsilon: float = 1.0) -> List[Tuple[float, float]]:
    """
    Ramer-Douglas-Peucker (RDP) recursive polygonal line simplification.
    Reduces irregular raster contour stair-stepping into clean rectilinear architectural facades.
    """
    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, (Real, np.number)):
        raise ValueError("epsilon must be a finite nonnegative number.")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be a finite nonnegative number.")

    try:
        pts = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("points must contain finite [x, y] pairs.") from exc
    if pts.size == 0:
        return []
    if pts.ndim != 2 or pts.shape[1] != 2 or not np.all(np.isfinite(pts)):
        raise ValueError("points must contain finite [x, y] pairs.")

    if len(points) <= 3:
        return points

    start, end = pts[0], pts[-1]
    line_vec = end - start
    line_len = float(np.hypot(line_vec[0], line_vec[1]))

    if line_len < 1e-9:
        dists = np.hypot(pts[:, 0] - start[0], pts[:, 1] - start[1])
    else:
        dists = np.abs(
            line_vec[1] * pts[:, 0] - line_vec[0] * pts[:, 1] + end[0] * start[1] - end[1] * start[0]
        ) / line_len

    idx = int(np.argmax(dists))
    max_dist = float(dists[idx])

    if max_dist > epsilon and 0 < idx < len(points) - 1:
        left = rdp_simplify(points[: idx + 1], epsilon)
        right = rdp_simplify(points[idx:], epsilon)
        return left[:-1] + right

    return [points[0], points[-1]]


def extract_lod1_buildings(
    dsm: np.ndarray,
    dtm: np.ndarray,
    gsd_m: float = 1.0,
    min_height_m: float = 2.5,
    min_area_m2: float = 20.0,
    simplify_epsilon_m: float = 1.0,
    max_buildings: int = 500
) -> List[Dict[str, Any]]:
    """
    Extracts 3D LoD-1 architectural building blocks from DSM and DTM surfaces.
    
    Hardened against:
      - Non-finite (NaN / Inf / -Inf) DSM or DTM grids
      - Non-positive or non-finite Ground Sample Distance (gsd_m)
      - Degenerate footprints or sub-resolution clusters
    """
    def finite_number(value, name, *, minimum=None, maximum=None):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, np.number)):
            raise ValueError(f"{name} must be a finite numeric value.")
        value = float(value)
        if not math.isfinite(value) or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
            bounds = []
            if minimum is not None:
                bounds.append(f">={minimum}")
            if maximum is not None:
                bounds.append(f"<={maximum}")
            suffix = f" ({', '.join(bounds)})" if bounds else ""
            raise ValueError(f"{name} must be finite{suffix}.")
        return value

    gsd_m = finite_number(gsd_m, "gsd_m", minimum=1e-6, maximum=100000.0)
    min_height_m = finite_number(min_height_m, "min_height_m", minimum=0.0, maximum=100000.0)
    min_area_m2 = finite_number(min_area_m2, "min_area_m2", minimum=0.0, maximum=1e12)
    simplify_epsilon_m = finite_number(simplify_epsilon_m, "simplify_epsilon_m", minimum=0.0, maximum=1e6)
    if (isinstance(max_buildings, (bool, np.bool_)) or not isinstance(max_buildings, (int, np.integer))
            or not 1 <= int(max_buildings) <= 100000):
        raise ValueError("max_buildings must be an integer between 1 and 100000.")
    max_buildings = int(max_buildings)

    if not isinstance(dsm, np.ndarray) or not isinstance(dtm, np.ndarray):
        raise TypeError("dsm and dtm must be numpy arrays.")

    if dsm.shape != dtm.shape or dsm.ndim != 2:
        raise ValueError("DSM and DTM must be 2D arrays with matching dimensions.")

    rows, cols = dsm.shape
    if rows < 4 or cols < 4:
        return []

    try:
        dsm_values = np.ma.getdata(np.ma.asarray(dsm, dtype=np.float64))
        dtm_values = np.ma.getdata(np.ma.asarray(dtm, dtype=np.float64))
        dsm_mask = np.ma.getmaskarray(np.ma.asarray(dsm, dtype=np.float64))
        dtm_mask = np.ma.getmaskarray(np.ma.asarray(dtm, dtype=np.float64))
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("dsm and dtm must contain numeric values.") from exc

    # Sanitise DSM and DTM grids against hidden masked values, NaN and Inf.
    finite_mask = (~dsm_mask & ~dtm_mask & np.isfinite(dsm_values) & np.isfinite(dtm_values))
    if not np.any(finite_mask):
        return []

    dsm_clean = np.where(finite_mask, dsm_values, 0.0)
    dtm_clean = np.where(finite_mask, dtm_values, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = dsm_clean - dtm_clean
    finite_mask &= np.isfinite(difference)
    ndsm = np.where(finite_mask, difference, 0.0)

    # Initial height thresholding
    building_mask = ((ndsm >= min_height_m) & finite_mask).astype(np.uint8)
    if not np.any(building_mask > 0):
        return []

    if HAS_CV2:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_OPEN, k)
        building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, k)
        # Closing may fill a missing-data hole, but cannot create observations.
        building_mask[~finite_mask] = 0

        # Distance-transform guided instance separation
        dist_transform = cv2.distanceTransform(building_mask, cv2.DIST_L2, 5)
        max_dist = float(np.max(dist_transform)) if np.any(dist_transform > 0) else 0.0

        if max_dist > 3.0:
            # Seed peaks as markers for watershed
            sure_fg = (dist_transform > 0.35 * max_dist).astype(np.uint8)
            unknown = (building_mask > 0) & (sure_fg == 0)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown] = 0

            # Gradient map for watershed boundary definition
            grad_x = cv2.Sobel(ndsm.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(ndsm.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.hypot(grad_x, grad_y)
            grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            grad_bgr = cv2.cvtColor(grad_norm, cv2.COLOR_GRAY2BGR)

            markers = cv2.watershed(grad_bgr, markers)
            labels = np.where((markers > 1) & (building_mask > 0) & finite_mask, markers - 1, 0)
        else:
            _, labels, _, _ = cv2.connectedComponentsWithStats(building_mask, connectivity=8)
    else:
        from scipy import ndimage
        labels, _ = ndimage.label(building_mask)

    pixel_area_m2 = gsd_m * gsd_m
    if not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise ValueError("gsd_m is too small or large for a finite pixel area.")
    min_area_px_value = min_area_m2 / pixel_area_m2
    if not math.isfinite(min_area_px_value) or min_area_px_value > np.iinfo(np.int64).max:
        raise ValueError("min_area_m2 and gsd_m do not produce a bounded pixel-area threshold.")
    min_area_px = int(math.ceil(min_area_px_value))
    epsilon_px = max(1.0, simplify_epsilon_m / max(gsd_m, 0.1))

    buildings = []

    # scipy returns foreground count while OpenCV includes background in its
    # count. Actual label IDs avoid dropping the last (or only) component.
    for label_idx in np.unique(labels[labels > 0]):
        component_mask = ((labels == label_idx) & finite_mask & (building_mask > 0)).astype(np.uint8)
        area_px = int(np.sum(component_mask))
        if area_px < min_area_px:
            continue

        if HAS_CV2:
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            pts_raw = [(float(pt[0][0]), float(pt[0][1])) for pt in cnt]
        else:
            y_indices, x_indices = np.where(component_mask)
            if len(x_indices) == 0:
                continue
            min_x, max_x = float(np.min(x_indices)), float(np.max(x_indices))
            min_y, max_y = float(np.min(y_indices)), float(np.max(y_indices))
            pts_raw = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

        if len(pts_raw) < 3:
            continue

        if pts_raw[0] != pts_raw[-1]:
            pts_raw.append(pts_raw[0])

        simplified_pts = rdp_simplify(pts_raw, epsilon=epsilon_px)
        if len(simplified_pts) < 3:
            simplified_pts = pts_raw

        footprint_ndsm = ndsm[component_mask > 0]
        footprint_dtm = dtm_clean[component_mask > 0]

        valid_ndsm = footprint_ndsm[np.isfinite(footprint_ndsm)]
        valid_dtm = footprint_dtm[np.isfinite(footprint_dtm)]

        if len(valid_ndsm) == 0 or len(valid_dtm) == 0:
            continue

        building_height = float(np.percentile(valid_ndsm, 90))
        base_elevation = float(np.median(valid_dtm))

        if not math.isfinite(building_height) or not math.isfinite(base_elevation):
            continue

        roof_elevation = base_elevation + building_height
        footprint_area_m2 = float(area_px * (gsd_m ** 2))

        comp_ys, comp_xs = np.where(component_mask)
        cy, cx = float(np.mean(comp_ys)), float(np.mean(comp_xs))

        webgl_footprint = []
        for x, y in simplified_pts:
            wx = (x / float(cols) - 0.5) * 2.0
            wy = -(y / float(rows) - 0.5) * 2.0
            webgl_footprint.append([round(wx, 4), round(wy, 4)])

        buildings.append({
            "building_id": f"BLDG-{len(buildings)+1:04d}",
            "centroid_pixel": [round(cx, 1), round(cy, 1)],
            "footprint_pixel_coords": [[round(x, 1), round(y, 1)] for x, y in simplified_pts],
            "webgl_footprint_coords": webgl_footprint,
            "base_elevation_m": round(base_elevation, 2),
            "roof_elevation_m": round(roof_elevation, 2),
            "height_agl_m": round(building_height, 2),
            "footprint_area_m2": round(footprint_area_m2, 1),
            "geometry_type": "LoD-1 Prism"
        })

    buildings.sort(key=lambda b: b["height_agl_m"], reverse=True)
    return buildings[:max_buildings]
