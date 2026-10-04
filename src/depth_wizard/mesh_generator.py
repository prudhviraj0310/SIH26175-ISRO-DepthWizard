"""
DepthWizard 3D Mesh & Analytical Geomorphology Generator
========================================================
Transforms 2D Digital Surface Models (DSM) and optical satellite imagery
into 3D textured terrain meshes ready for Three.js WebGL flythrough rendering.

Features:
- Edge-preserving planarization (eliminates rooftop needle spikes while maintaining sharp vertical facades)
- Analytical Hillshading (NW azimuth 315 deg, altitude 45 deg)
- CartoDEM Hypsometric Elevation Colormapping (standard USGS/ISRO ramp)
- Blended Topographic Relief Texture (Hypsometric x Hillshade)
- Geomorphological Slope Angle Classification (<5, 5-15, 15-30, >30 deg)
- Ortho-Hillshade Hybrid (optical satellite imagery modulated by physical shading)
- Live Error Difference Map (|DSM_AI - DSM_LiDAR|) in survey-grade color bands
- Physical grid coordinate arrays for real-time cursor probing
"""

import io
import base64
from collections.abc import Mapping
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
from scipy import ndimage

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class MeshGenerator:
    def __init__(self, target_grid_size: int = 128):
        if (isinstance(target_grid_size, (bool, np.bool_))
                or not isinstance(target_grid_size, (int, np.integer))
                or not 2 <= int(target_grid_size) <= 512):
            raise ValueError("target_grid_size must be an integer between 2 and 512.")
        self.target_grid_size = int(target_grid_size)

    @staticmethod
    def _finite_grid(value, name):
        try:
            masked = np.ma.asarray(value, dtype=np.float64)
            data = np.ma.getdata(masked)
            hidden_mask = np.ma.getmaskarray(masked)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a numeric 2D grid.") from exc
        if data.ndim != 2 or min(data.shape) < 2 or max(data.shape) > 2048:
            raise ValueError(f"{name} must be a 2D grid with sides between 2 and 2048.")
        if (np.any(hidden_mask) or not np.all(np.isfinite(data))
                or np.any(np.abs(data) > 1e6) or np.any(np.isin(data, [-9999, -32768]))):
            raise ValueError(f"{name} contains nonfinite or NODATA support.")
        return np.asarray(data, dtype=np.float32)

    @staticmethod
    def _rgb_array(value):
        try:
            masked = np.ma.asarray(value)
            data = np.ma.getdata(masked)
            hidden_mask = np.ma.getmaskarray(masked)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("rgb_image must be a finite HxWx3 image.") from exc
        if data.ndim != 3 or data.shape[2] != 3 or min(data.shape[:2]) < 2 or max(data.shape[:2]) > 2048:
            raise ValueError("rgb_image must be a HxWx3 image with sides between 2 and 2048.")
        if np.any(hidden_mask):
            raise ValueError("rgb_image contains masked or nonfinite radiometry.")
        try:
            numeric = np.asarray(data, dtype=np.float64)
            if not np.all(np.isfinite(numeric)):
                raise ValueError("rgb_image contains masked or nonfinite radiometry.")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("rgb_image must contain numeric finite radiometry.") from exc
        if data.dtype != np.uint8:
            data = np.clip(numeric, 0, 255).astype(np.uint8)
        return np.asarray(data)

    def _encode_image(self, arr: np.ndarray, quality: int = 85) -> str:
        pil_img = Image.fromarray(arr)
        if pil_img.size != (512, 512):
            pil_img = pil_img.resize((512, 512), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate_mesh_payload(
        self,
        dsm: np.ndarray,
        rgb_image: np.ndarray,
        stats: Dict[str, Any],
        ground_truth_dsm: Optional[np.ndarray] = None,
        dtm: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Processes DSM and RGB into a high-fidelity WebGL payload:
        - Applies bilateral planarization to eliminate single-pixel rooftop needle spikes
        - Downsamples grid to target_grid_size x target_grid_size for responsive Three.js rendering
        - Generates 512x512 analytical hillshade, hypsometric tint, slope map, and error difference map
        - Encodes all layers as Base64 JPEG data URLs for instant client-side switching
        """
        if not isinstance(stats, Mapping):
            raise ValueError("stats must be a mapping of verified mesh metadata.")
        dsm = self._finite_grid(dsm, "dsm")
        rgb_image = self._rgb_array(rgb_image)
        if rgb_image.shape[:2] != dsm.shape:
            raise ValueError("rgb_image and dsm must have matching spatial dimensions.")
        if ground_truth_dsm is not None:
            ground_truth_dsm = self._finite_grid(ground_truth_dsm, "ground_truth_dsm")
            if ground_truth_dsm.shape != dsm.shape:
                raise ValueError("ground_truth_dsm and dsm must have matching dimensions.")
        if dtm is not None:
            dtm = self._finite_grid(dtm, "dtm")
            if dtm.shape != dsm.shape:
                raise ValueError("dtm and dsm must have matching dimensions.")

        gsd_value = stats.get("ground_sample_dist_m", stats.get("gsd_m", 0.5))
        if isinstance(gsd_value, (bool, np.bool_)):
            raise ValueError("ground_sample_dist_m must be numeric, not boolean.")
        try:
            gsd = float(gsd_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("ground_sample_dist_m must be finite and positive.") from exc
        if not np.isfinite(gsd) or not 1e-6 <= gsd <= 100000:
            raise ValueError("ground_sample_dist_m must be finite and positive.")

        orig_rows, orig_cols = dsm.shape
        target_s = self.target_grid_size

        # 1. Edge-preserving planarization to eliminate rooftop needle spikes
        if HAS_CV2:
            smooth_dsm = cv2.bilateralFilter(dsm.astype(np.float32), d=5, sigmaColor=3.0, sigmaSpace=3.0)
        else:
            smooth_dsm = ndimage.median_filter(dsm.astype(np.float32), size=3)

        # Subsample planarized DSM to target grid size for 3D vertex mesh
        row_indices = np.linspace(0, orig_rows - 1, target_s).astype(int)
        col_indices = np.linspace(0, orig_cols - 1, target_s).astype(int)
        sub_dsm = smooth_dsm[np.ix_(row_indices, col_indices)]

        # Optical satellite texture
        texture_base64 = self._encode_image(rgb_image)

        # Elevation bounds
        min_z = float(np.min(dsm))
        max_z = float(np.max(dsm))
        z_range = max(1.0, max_z - min_z)
        normalized_z = np.clip((sub_dsm - min_z) / z_range, 0.0, 1.0)

        # Full resolution 512x512 DSM for crisp hillshade & textures
        full_dsm_img = Image.fromarray(smooth_dsm.astype(np.float32))
        full_dsm_512 = np.array(full_dsm_img.resize((512, 512), Image.Resampling.BILINEAR))
        # Geometry smoothing is a display heuristic, not a modification of the
        # source surface used to describe slope. It flattens even planar edges.
        slope_surface_512 = np.array(Image.fromarray(dsm).resize((512, 512), Image.Resampling.BILINEAR))

        # PIL resizes pixel-centred rasters: neighbouring output centres are
        # orig_cols/512 and orig_rows/512 source pixels apart. A rectangular
        # image therefore needs different x/y spacings after square resampling.
        spacing_x_m = gsd * orig_cols / full_dsm_512.shape[1]
        spacing_y_m = gsd * orig_rows / full_dsm_512.shape[0]
        dz_dx = np.gradient(slope_surface_512, spacing_x_m, axis=1)
        dz_dy = np.gradient(slope_surface_512, spacing_y_m, axis=0)

        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)
        aspect_rad = np.arctan2(-dz_dy, dz_dx)
        aspect_rad = np.where(aspect_rad < 0, 2 * np.pi + aspect_rad, aspect_rad)

        # 2. Analytical Hillshade (Standard Cartographic NW: az=315 deg, alt=45 deg)
        sun_alt = np.radians(45.0)
        sun_az = np.radians(315.0)
        hs = 255.0 * ((np.sin(sun_alt) * np.cos(slope_rad)) + 
                      (np.cos(sun_alt) * np.sin(slope_rad) * np.cos(sun_az - aspect_rad)))
        hs = np.clip(hs, 0, 255).astype(np.uint8)
        hillshade_url = self._encode_image(np.stack([hs, hs, hs], axis=-1))

        # 3. Hypsometric Tint (CartoDEM / ISRO standard elevation colormap)
        norm_512 = np.clip((full_dsm_512 - min_z) / z_range, 0.0, 1.0)
        hypso_rgb = np.zeros((512, 512, 3), dtype=np.uint8)

        c0 = np.array([30, 80, 115])   # Deep valley / basin (slate teal)
        c1 = np.array([45, 135, 80])   # Lowland plains (rich forest green)
        c2 = np.array([135, 170, 75])  # Gentle midlands (olive green)
        c3 = np.array([215, 175, 70])  # Terraces & foothills (warm ochre)
        c4 = np.array([185, 95, 45])   # Upper slopes & ridges (terracotta sienna)
        c5 = np.array([115, 75, 60])   # Summit / structural peak (warm mountain rock)

        m1 = norm_512 < 0.2
        t1 = norm_512[m1] / 0.2
        hypso_rgb[m1] = (c0[None, :] * (1 - t1[:, None]) + c1[None, :] * t1[:, None]).astype(np.uint8)

        m2 = (norm_512 >= 0.2) & (norm_512 < 0.45)
        t2 = (norm_512[m2] - 0.2) / 0.25
        hypso_rgb[m2] = (c1[None, :] * (1 - t2[:, None]) + c2[None, :] * t2[:, None]).astype(np.uint8)

        m3 = (norm_512 >= 0.45) & (norm_512 < 0.7)
        t3 = (norm_512[m3] - 0.45) / 0.25
        hypso_rgb[m3] = (c2[None, :] * (1 - t3[:, None]) + c3[None, :] * t3[:, None]).astype(np.uint8)

        m4 = (norm_512 >= 0.7) & (norm_512 < 0.9)
        t4 = (norm_512[m4] - 0.7) / 0.2
        hypso_rgb[m4] = (c3[None, :] * (1 - t4[:, None]) + c4[None, :] * t4[:, None]).astype(np.uint8)

        m5 = norm_512 >= 0.9
        t5 = (norm_512[m5] - 0.9) / 0.1
        hypso_rgb[m5] = (c4[None, :] * (1 - t5[:, None]) + c5[None, :] * t5[:, None]).astype(np.uint8)

        hypsometric_url = self._encode_image(hypso_rgb)

        # 4. Blended Relief (Hypsometric x Hillshade) -> Gives dramatic 3D slope depth!
        hs_factor = (hs.astype(np.float32) / 255.0)[:, :, None] ** 0.85
        blended_relief = np.clip(hypso_rgb.astype(np.float32) * hs_factor * 1.15, 0, 255).astype(np.uint8)
        relief_url = self._encode_image(blended_relief)

        # 5. Slope Classification Map (<5 flat, 5-15 gentle, 15-30 moderate, >30 steep)
        slope_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        slope_rgb[slope_deg < 5.0] = [46, 204, 113]                     # Green (Safe/Flat)
        slope_rgb[(slope_deg >= 5.0) & (slope_deg < 15.0)] = [241, 196, 15]  # Yellow (Gentle)
        slope_rgb[(slope_deg >= 15.0) & (slope_deg < 30.0)] = [230, 126, 34] # Orange (Moderate)
        slope_rgb[slope_deg >= 30.0] = [231, 76, 60]                    # Red (Steep Escarpment)
        slope_shaded = np.clip(slope_rgb.astype(np.float32) * hs_factor * 1.1, 0, 255).astype(np.uint8)
        slope_url = self._encode_image(slope_shaded)

        # 6. Ortho + Hillshade Hybrid
        pil_rgb = Image.fromarray(rgb_image).resize((512, 512), Image.Resampling.LANCZOS)
        rgb_arr = np.array(pil_rgb)
        ortho_shaded = np.clip(rgb_arr.astype(np.float32) * (hs_factor * 1.05 + 0.05), 0, 255).astype(np.uint8)
        ortho_hs_url = self._encode_image(ortho_shaded)

        # 7. Reference-array difference when a complete reference is supplied.
        if ground_truth_dsm is not None:
            full_gt_img = Image.fromarray(ground_truth_dsm.astype(np.float32))
            full_gt_512 = np.array(full_gt_img.resize((512, 512), Image.Resampling.BILINEAR))
            error_diff = np.abs(full_dsm_512 - full_gt_512)
        else:
            error_diff = np.abs(full_dsm_512 - ndimage.gaussian_filter(full_dsm_512, sigma=2.0))

        error_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        error_rgb[error_diff < 1.5] = [46, 204, 113]                        # Low surface-unit difference
        error_rgb[(error_diff >= 1.5) & (error_diff < 3.0)] = [241, 196, 15] # Intermediate surface-unit difference
        error_rgb[error_diff >= 3.0] = [231, 76, 60]                       # Red (>3.0m Discrepancy)
        error_shaded = np.clip(error_rgb.astype(np.float32) * hs_factor * 1.1, 0, 255).astype(np.uint8)
        error_url = self._encode_image(error_shaded)

        # Subsampled slope degrees for cursor probe
        sub_slope = slope_deg[np.ix_(np.linspace(0, 511, target_s).astype(int), np.linspace(0, 511, target_s).astype(int))]

        # 6. LoD-1 Architectural Building Block Extrusion (Experimental 3D Visualization)
        lod1_buildings = []
        lod1_status = "SUCCESS"
        lod1_error = None
        lod1_dtm_source = "UNAVAILABLE"
        lod1_dtm_status = "UNAVAILABLE"
        lod1_dtm_authoritative = False
        terrain_provenance = stats.get("terrain_provenance", {})
        if not isinstance(terrain_provenance, Mapping):
            terrain_provenance = {}
        try:
            try:
                from depth_wizard.lod1_extractor import extract_lod1_buildings
            except ImportError:
                from src.depth_wizard.lod1_extractor import extract_lod1_buildings
            if dtm is None:
                # This is a visualization fallback, not an observed bare-earth DTM.
                if HAS_CV2:
                    k_dtm = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                    bare_earth_dtm = cv2.morphologyEx(smooth_dsm.astype(np.float32), cv2.MORPH_OPEN, k_dtm)
                else:
                    bare_earth_dtm = ndimage.grey_opening(smooth_dsm.astype(np.float32), size=(15, 15))
                lod1_dtm_source = "MORPHOLOGICAL_FALLBACK_FROM_SURFACE"
                lod1_dtm_status = "UNVERIFIED_MORPHOLOGICAL_FALLBACK"
            else:
                bare_earth_dtm = dtm
                lod1_dtm_source = stats.get("dtm_source", "SUPPLIED_TERRAIN_REFERENCE")
                lod1_dtm_authoritative = bool(
                    stats.get("terrain_reference_kind") == "DTM"
                    and stats.get("vertical_reference_status") == "VERIFIED"
                    and terrain_provenance.get("bare_earth_certified") is True
                    and terrain_provenance.get("datum_verified") is True
                )
                lod1_dtm_status = "AUTHORITATIVE_VERIFIED_DTM" if lod1_dtm_authoritative else "SUPPLIED_UNVERIFIED_REFERENCE"

            lod1_buildings = extract_lod1_buildings(smooth_dsm, bare_earth_dtm, gsd_m=gsd)
            lod1_status = "SUCCESS" if lod1_buildings else "EMPTY"
        except Exception as e:
            lod1_buildings = []
            lod1_status = "FAILED"
            lod1_error = str(e)
            lod1_dtm_source = "UNAVAILABLE"
            lod1_dtm_status = "UNAVAILABLE"

        return {
            "grid_size": target_s,
            "min_elevation_m": round(min_z, 2),
            "max_elevation_m": round(max_z, 2),
            "elevation_range_m": round(z_range, 2),
            "metric_heights": [round(float(h), 2) for h in sub_dsm.flatten()],
            "normalized_z": [round(float(z), 4) for z in normalized_z.flatten()],
            "slope_degrees": [round(float(s), 1) for s in sub_slope.flatten()],
            "mean_slope_deg": round(float(np.mean(slope_deg)), 1),
            "max_slope_deg": round(float(np.max(slope_deg)), 1),
            "flat_percentage": round(float(np.mean(slope_deg < 5.0) * 100), 1),
            "steep_percentage": round(float(np.mean(slope_deg >= 30.0) * 100), 1),
            "texture_data_url": texture_base64,
            "hillshade_texture_url": hillshade_url,
            "hypsometric_texture_url": hypsometric_url,
            "relief_texture_url": relief_url,
            "slope_texture_url": slope_url,
            "ortho_hillshade_url": ortho_hs_url,
            "error_texture_url": error_url,
            "stats": stats,
            "source_grid_dimensions": [int(orig_rows), int(orig_cols)],
            "ground_sample_dist_m": gsd,
            "slope_grid_spacing_m": [spacing_y_m, spacing_x_m],
            "lod1_buildings": lod1_buildings,
            "lod1_building_count": len(lod1_buildings),
            "lod1_status": lod1_status,
            "lod1_error": lod1_error,
            "lod1_dtm_source": lod1_dtm_source,
            "lod1_dtm_status": lod1_dtm_status,
            "lod1_dtm_authoritative": lod1_dtm_authoritative,
            "reference_difference_available": ground_truth_dsm is not None,
        }
