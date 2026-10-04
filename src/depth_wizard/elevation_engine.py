"""Relative appearance priors, explicit height assumptions, and terrain geometry.

Monocular appearance does not supply a metric height scale by itself. Source
terrain is retained and DSM = terrain reference + estimated structural height.
Geometric screens are not flight clearance or geotechnical certification.
"""

import os
import io
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from scipy import ndimage

try:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    HAS_TORCH_TRANSFORMERS = False

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from src.depth_wizard.srtm_provider import SRTMElevationProvider
    HAS_SRTM_PROVIDER = True
except ImportError:
    try:
        from depth_wizard.srtm_provider import SRTMElevationProvider
        HAS_SRTM_PROVIDER = True
    except ImportError:
        HAS_SRTM_PROVIDER = False


def _inference_mode(func):
    """Safe wrapper for torch.inference_mode() that works without torch."""
    if HAS_TORCH_TRANSFORMERS:
        import torch
        return torch.inference_mode()(func)
    return func


class DepthAnythingV2Backbone:
    """
    Depth Anything V2 Monocular Foundation Model Loader.
    Provides fast inference on Apple Silicon (MPS), NVIDIA CUDA, or CPU.
    """
    _instance = None
    _model = None
    _processor = None
    _device = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        self.model_id = model_id
        self.load_status = "UNAVAILABLE"
        self.load_error = None
        self._model = None
        self._processor = None
        if not HAS_TORCH_TRANSFORMERS:
            self.load_error = "torch/transformers dependencies are unavailable."
            return

        try:
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"

            # Reproducible offline loading: a missing cache is an unavailable
            # model, never an implicit download or a neural-performance claim.
            self._processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
            self._model = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=True).to(self._device)
            self._model.eval()
            self.load_status = "AVAILABLE_CACHED_MODEL"
            print(f"✓ Depth Anything V2 loaded on {self._device.upper()}")
        except Exception as e:
            self.load_error = f"{type(e).__name__}: cached Depth Anything V2 model could not be loaded."
            self._model = None

    @_inference_mode
    def infer(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Runs Depth Anything V2 inference on RGB imagery.
        For large images (>1024px), uses tiled inference with Gaussian feather
        blending to avoid edge artifacts while maintaining GPU memory efficiency.
        Returns normalized relative height prior in [0, 1].
        """
        if self._model is None or self._processor is None:
            return None

        h, w = rgb_image.shape[:2]

        # For large satellite scenes, use tiled inference with overlap blending
        TILE_THRESHOLD = 1024
        if max(h, w) > TILE_THRESHOLD:
            return self._tiled_infer(rgb_image, tile_size=512, overlap=64)

        pil_img = Image.fromarray(rgb_image)
        inputs = self._processor(images=pil_img, return_tensors="pt").to(self._device)
        outputs = self._model(**inputs)
        raw_depth = outputs.predicted_depth.squeeze().cpu().numpy()

        # Resize back to target input resolution
        depth_resized = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # In monocular perspective, distance to sensor is inverted to height (closer = taller)
        height_prior = np.max(depth_resized) - depth_resized

        # Robust percentile normalization
        p_min, p_max = np.percentile(height_prior, [3, 97])
        norm_h = np.clip((height_prior - p_min) / (p_max - p_min + 1e-8), 0.0, 1.0)
        return norm_h.astype(np.float32)

    @_inference_mode
    def _tiled_infer(self, rgb_image: np.ndarray, tile_size: int = 512, overlap: int = 64) -> np.ndarray:
        """
        Windowed tiled inference with Gaussian feather blending.
        Splits large satellite images into overlapping tiles, runs DAv2 on each,
        and blends results using a 2D Gaussian weight kernel to eliminate seams.
        
        Reference: Ronneberger et al., U-Net (2015) - overlap-tile strategy
        """
        h, w = rgb_image.shape[:2]
        stride = tile_size - overlap
        
        # Output accumulator and weight map
        depth_accum = np.zeros((h, w), dtype=np.float64)
        weight_accum = np.zeros((h, w), dtype=np.float64)
        
        # 2D Gaussian blending kernel (Hann-like feather window)
        sigma = tile_size / 4.0
        ax = np.arange(tile_size) - tile_size / 2.0
        xx, yy = np.meshgrid(ax, ax)
        gaussian_kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        for y0 in range(0, h, stride):
            for x0 in range(0, w, stride):
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)
                tile = rgb_image[y0:y1, x0:x1]
                
                # Pad if tile is smaller than expected
                th, tw = tile.shape[:2]
                if th < tile_size or tw < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    padded[:th, :tw] = tile
                    tile = padded
                
                # Run inference on single tile
                pil_tile = Image.fromarray(tile)
                inputs = self._processor(images=pil_tile, return_tensors="pt").to(self._device)
                outputs = self._model(**inputs)
                tile_depth = outputs.predicted_depth.squeeze().cpu().numpy()
                tile_depth = cv2.resize(tile_depth, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                
                # Invert: depth-to-height
                tile_height = np.max(tile_depth) - tile_depth
                
                # Crop kernel and result to actual tile dimensions
                kernel_crop = gaussian_kernel[:th, :tw]
                tile_crop = tile_height[:th, :tw]
                
                depth_accum[y0:y1, x0:x1] += tile_crop * kernel_crop
                weight_accum[y0:y1, x0:x1] += kernel_crop
        
        # Normalize by accumulated weights
        blended = depth_accum / (weight_accum + 1e-8)
        
        # Robust percentile normalization
        p_min, p_max = np.percentile(blended, [3, 97])
        norm_h = np.clip((blended - p_min) / (p_max - p_min + 1e-8), 0.0, 1.0)
        return norm_h.astype(np.float32)


def robust_affine_calibration_irls(
    relative_values: np.ndarray,
    metric_anchors: np.ndarray,
    weights: Optional[np.ndarray] = None,
    huber_delta: float = 1.5,
    max_iter: int = 100,
    tol: float = 1e-6,
    require_positive_scale: bool = True
) -> Dict[str, Any]:
    """Fit y = scale*x + offset to independent observations using Huber IRLS.

    Zero-weight and nonfinite x/y samples supply no support. Invalid weights,
    degeneracy, poor numerical conditioning, and nonconvergence are explicit;
    a rejected fit never acquires a fabricated positive scale or zero error.
    The caller must establish observation independence and physical semantics.
    """
    result = {"scale": None, "offset": None, "rmse_m": None, "nmad_m": None,
              "converged": False, "iterations": 0, "valid_sample_count": 0,
              "discarded_sample_count": 0, "rank": 0, "condition_number": None,
              "status": "INVALID_INPUT", "failure_reason": None}

    def reject(reason, status="INVALID_INPUT"):
        return {**result, "converged": False, "status": status, "failure_reason": reason}

    try:
        x = np.ma.asarray(relative_values, dtype=np.float64).filled(np.nan).reshape(-1)
        y = np.ma.asarray(metric_anchors, dtype=np.float64).filled(np.nan).reshape(-1)
        w = np.ones_like(x) if weights is None else np.ma.asarray(weights, dtype=np.float64).filled(np.nan).reshape(-1)
        valid_params = (np.isfinite(huber_delta) and huber_delta > 0 and np.isfinite(tol) and tol > 0
                        and isinstance(max_iter, (int, np.integer)) and not isinstance(max_iter, bool)
                        and 1 <= max_iter <= 1000)
    except (TypeError, ValueError, OverflowError):
        return reject("Inputs and solver parameters must be numeric.")
    if not valid_params or x.size != y.size or w.size != x.size:
        return reject("Sample/weight lengths must match and solver parameters must be positive and bounded.")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        return reject("Weights must be finite and nonnegative.", "INVALID_WEIGHTS")
    valid = np.isfinite(x) & np.isfinite(y) & (w > 0)
    result.update(valid_sample_count=int(np.sum(valid)), discarded_sample_count=int(np.sum(~valid)))
    x, y, w = x[valid], y[valid], w[valid]
    if x.size < 2:
        return reject("At least two finite positive-weight observations are required.", "INSUFFICIENT_SUPPORT")
    w = w / np.max(w)
    with np.errstate(over="ignore", invalid="ignore"):
        x_center = float(np.average(x, weights=w))
        x_spread = float(np.max(np.abs(x - x_center)))
    if not np.isfinite(x_center) or not np.isfinite(x_spread):
        return reject("Control coordinates cannot be centred with finite precision.", "ILL_CONDITIONED")
    if x_spread <= 100 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(x)))):
        result["rank"] = 1
        return reject("Relative observations have no numerically resolvable variation.", "RANK_DEFICIENT")
    # Centring/rescaling avoids large coordinate offsets destroying precision.
    z = (x - x_center) / x_spread
    A = np.column_stack([z, np.ones_like(z)])

    def solve(effective_weights):
        Aw = A * np.sqrt(effective_weights[:, None])
        yw = y * np.sqrt(effective_weights)
        p, _, rank, singular = np.linalg.lstsq(Aw, yw, rcond=None)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
        result["rank"] = int(rank)
        result["condition_number"] = condition if np.isfinite(condition) else None
        if rank < 2 or not np.isfinite(condition) or condition > 1e8 or not np.all(np.isfinite(p)):
            raise ValueError("Weighted control design is rank deficient or ill conditioned.")
        return p

    try:
        p = solve(w)
        for iteration in range(max_iter):
            result["iterations"] = iteration + 1
            residuals = y - A @ p
            huber_w = np.minimum(1.0, huber_delta / np.maximum(np.abs(residuals), np.finfo(float).tiny))
            p_new = solve(w * huber_w)
            if np.max(np.abs(p_new - p)) <= tol * (1.0 + np.max(np.abs(p))):
                p = p_new
                result["converged"] = True
                break
            p = p_new
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return reject("Weighted control design cannot support a stable affine fit.", "ILL_CONDITIONED")
    scale = float(p[0] / x_spread)
    offset = float(p[1] - scale * x_center)
    if not np.isfinite(scale) or not np.isfinite(offset):
        return reject("Affine parameters are not finite.", "ILL_CONDITIONED")
    signal_tolerance = 100 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(y))))
    if require_positive_scale and p[0] <= signal_tolerance:
        result["converged"] = False
        return reject("A positive physical height scale is not supported.", "NONPOSITIVE_SCALE")
    residuals = y - A @ p
    with np.errstate(over="ignore", invalid="ignore"):
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        nmad = float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))
    if not np.isfinite(rmse) or not np.isfinite(nmad):
        result["converged"] = False
        return reject("Fit residual statistics are nonfinite.", "ILL_CONDITIONED")
    result.update(scale=scale, offset=offset, rmse_m=rmse, nmad_m=nmad,
                  inlier_count=int(np.sum(np.abs(residuals) <= huber_delta)),
                  relative_range=[float(np.min(x)), float(np.max(x))],
                  status="FITTED" if result["converged"] else "NOT_CONVERGED",
                  failure_reason=None if result["converged"] else "Iteration limit reached.")
    return result


class ElevationEngine:
    """
    Scientific Elevation Extraction and Scale-Calibration Pipeline.
    """
    def __init__(self):
        self.dav2 = DepthAnythingV2Backbone.get_instance()
        self.last_inference_provenance = {"mode": "NOT_RUN", "metric_accuracy_validated": False}

    def extract_relative_depth(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Extract an unvalidated relative appearance prior in [0, 1].
        Model availability and the non-neural heuristic path are disclosed.
        Detrending, brightness and roof heuristics are not photogrammetry.
        """
        rgb_image = np.asarray(rgb_image)
        if rgb_image.ndim == 2:
            rgb_image = np.stack([rgb_image] * 3, axis=-1)
        if (rgb_image.ndim != 3 or rgb_image.shape[2] != 3 or min(rgb_image.shape[:2]) < 2
                or max(rgb_image.shape[:2]) > 2048 or not np.all(np.isfinite(rgb_image))):
            raise ValueError("RGB input must be a finite image of shape HxWx3 with sides between 2 and 2048.")
        backbone = getattr(self, "dav2", None)
        self.last_inference_provenance = {
            "mode": "APPEARANCE_HEURISTIC", "neural_model_used": False,
            "requested_model": getattr(backbone, "model_id", "depth-anything/Depth-Anything-V2-Small-hf"),
            "model_status": getattr(backbone, "load_status", "UNAVAILABLE"),
            "model_error": getattr(backbone, "load_error", "No loaded backbone." if backbone is None else None),
            "metric_accuracy_validated": False,
            "limitations": ["Brightness, edges and roof masks are not measured structural heights.",
                             "Metric scale and neural accuracy have not been established."],
        }

        gray = (0.299 * rgb_image[:, :, 0] + 0.587 * rgb_image[:, :, 1] + 0.114 * rgb_image[:, :, 2]).astype(np.float32)
        norm_gray = (gray - np.min(gray)) / (np.max(gray) - np.min(gray) + 1e-8)

        # 1. Multi-scale structural prominence (building edges & boundaries)
        grad_x = ndimage.sobel(norm_gray, axis=1)
        grad_y = ndimage.sobel(norm_gray, axis=0)
        edge_mag = np.hypot(grad_x, grad_y)

        blur_fine = ndimage.gaussian_filter(norm_gray, sigma=1.2)
        blur_coarse = ndimage.gaussian_filter(norm_gray, sigma=8.0)
        structural_prominence = np.maximum(0.0, blur_fine - blur_coarse)

        # High-contrast rooftop detection (institutional campuses, industrial parks)
        is_roof = (gray > 195).astype(np.float32)
        roof_area_frac = float(np.mean(is_roof))
        if 0.02 < roof_area_frac < 0.45:
            opened_roof = ndimage.binary_opening(is_roof, structure=np.ones((5, 5)))
            solid_roof = ndimage.binary_fill_holes(opened_roof).astype(np.float32)
        else:
            solid_roof = np.zeros_like(norm_gray)

        # 2. Run Foundation Deep Learning Model (Depth Anything V2)
        if HAS_TORCH_TRANSFORMERS and backbone is not None:
            try:
                ai_height = backbone.infer(rgb_image)
                if ai_height is not None:
                    if ai_height.shape != norm_gray.shape or not np.all(np.isfinite(ai_height)):
                        raise ValueError("Depth model returned an invalid relative grid.")
                    h, w = ai_height.shape
                    # Orthographic Detrending: remove ground-level perspective tilt
                    y_grid, x_grid = np.mgrid[0:h, 0:w]
                    ground_mask = structural_prominence < np.percentile(structural_prominence, 60)
                    detrending_applied = False
                    if np.sum(ground_mask) > 100:
                        A = np.column_stack([x_grid[ground_mask] / w, y_grid[ground_mask] / h, np.ones(np.sum(ground_mask))])
                        coeffs, _, _, _ = np.linalg.lstsq(A, ai_height[ground_mask], rcond=None)
                        tilt_plane = coeffs[0] * (x_grid / w) + coeffs[1] * (y_grid / h) + coeffs[2]
                        detrended = np.maximum(0.0, ai_height - tilt_plane)
                        detrended = (detrended - detrended.min()) / (detrended.max() - detrended.min() + 1e-8)
                        detrending_applied = True
                    else:
                        detrended = ai_height

                    if np.any(solid_roof > 0.5):
                        rel_depth = 0.70 * solid_roof + 0.20 * detrended + 0.10 * edge_mag
                    else:
                        rel_depth = (
                            0.45 * detrended +
                            0.35 * structural_prominence +
                            0.15 * norm_gray +
                            0.05 * edge_mag
                        )
                    rel_depth = (rel_depth - np.min(rel_depth)) / (np.max(rel_depth) - np.min(rel_depth) + 1e-8)
                    self.last_inference_provenance.update(
                        mode="NEURAL_RELATIVE_PLUS_APPEARANCE_HEURISTICS", neural_model_used=True,
                        relative_orientation="MAX_MINUS_MODEL_OUTPUT_ASSUMPTION", detrending_applied=detrending_applied)
                    self._last_relative_prior = rel_depth.astype(np.float32)
                    return self._last_relative_prior
                if self.last_inference_provenance["model_status"] == "AVAILABLE_CACHED_MODEL":
                    self.last_inference_provenance.update(model_status="OUTPUT_UNAVAILABLE", model_error="No neural prior was returned.")
            except Exception as e:
                self.last_inference_provenance.update(model_status="INFERENCE_FAILED", model_error=type(e).__name__)

        # 3. Non-neural appearance heuristic (unvalidated as height).
        if np.any(solid_roof > 0.5):
            rel_depth = 0.75 * solid_roof + 0.15 * structural_prominence + 0.10 * edge_mag
        else:
            rel_depth = (
                0.55 * norm_gray +
                0.30 * structural_prominence +
                0.15 * edge_mag
            )
        rel_depth = (rel_depth - np.min(rel_depth)) / (np.max(rel_depth) - np.min(rel_depth) + 1e-8)
        self._last_relative_prior = rel_depth.astype(np.float32)
        return self._last_relative_prior



    def calibrate_to_absolute_dsm(
        self,
        rel_depth: np.ndarray,
        base_srtm_elevation_m: float,
        terrain_gradient_m: float = 0.0,
        max_structural_height_m: float = 45.0,
        solar_elevation_deg: float = 58.5,
        gsd_m: float = 0.6,
        geo_bounds: list = None,
        is_georeferenced: bool = True,
        use_robust_irls: bool = True,
        terrain_grid: Optional[np.ndarray] = None,
        terrain_provenance: Optional[Dict[str, Any]] = None,
        independent_height_controls: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compose a surface without fitting away independently known relief.

        ``dtm`` is a compatibility name for the terrain *reference*. A public
        DSM is not bare earth. Unless supported independent AGL observations
        are supplied, ``max_structural_height_m`` is an assumed scale, not a
        photogrammetric calibration. GSD and solar angle cannot establish it.

        Optional controls contain pixel_x, pixel_y, heights_agl_m, source,
        independent=True and measurement_type (SURVEYED_AGL, LIDAR_AGL or
        KNOWN_HEIGHT_AGL). Evaluation/synthetic labels are not calibration
        controls. Absolute metric status additionally requires independently
        validated, aligned bare-earth terrain with a verified vertical datum.
        """
        tracked_prior = getattr(self, "_last_relative_prior", None)
        inference_provenance = (dict(self.last_inference_provenance) if tracked_prior is not None and rel_depth is tracked_prior
                                else {"mode": "EXTERNAL_RELATIVE_PRIOR", "metric_accuracy_validated": False})
        rel_depth = np.ma.asarray(rel_depth, dtype=np.float64).filled(np.nan)
        if rel_depth.ndim != 2 or not rel_depth.size or max(rel_depth.shape) > 2048:
            raise ValueError("Relative height must be a nonempty 2D grid no larger than 2048 pixels per side.")
        for value, name, bound in [(gsd_m, "gsd_m", 100000),
                                   (max_structural_height_m, "max_structural_height_m", 1000)]:
            if not np.isfinite(value) or not 0 < value <= bound:
                raise ValueError(f"{name} must be finite, positive and bounded.")
        if not np.isfinite(base_srtm_elevation_m) or not -500 <= base_srtm_elevation_m <= 10000:
            raise ValueError("Base elevation must be a finite plausible reference.")
        if not np.isfinite(terrain_gradient_m) or abs(terrain_gradient_m) > 10000:
            raise ValueError("Terrain gradient must be finite and bounded.")
        finite_relative = np.isfinite(rel_depth)
        if np.any((rel_depth[finite_relative] < 0) | (rel_depth[finite_relative] > 1)):
            raise ValueError("Relative height values must lie in [0, 1].")
        rows, cols = rel_depth.shape
        bounds = None
        if geo_bounds is not None:
            bounds = np.asarray(geo_bounds, dtype=np.float64)
            if (bounds.shape != (4,) or not np.all(np.isfinite(bounds))
                    or not -180 <= bounds[0] < bounds[2] <= 180
                    or not -90 <= bounds[1] < bounds[3] <= 90):
                raise ValueError("Calibration bounds must be finite WGS84 [west, south, east, north].")
            bounds = bounds.tolist()

        dtm = None
        provenance = dict(terrain_provenance or {})
        requested_source_failure = None
        if terrain_grid is not None:
            dtm = np.ma.asarray(terrain_grid, dtype=np.float64).filled(np.nan)
        elif bounds is not None and is_georeferenced and HAS_SRTM_PROVIDER:
            try:
                record = SRTMElevationProvider.get_elevation_grid(
                    bounds=bounds, grid_rows=rows, grid_cols=cols, return_metadata=True)
                if isinstance(record, dict):
                    dtm = np.ma.asarray(record["grid"], dtype=np.float64).filled(np.nan)
                    provenance = dict(record.get("provenance", {k: v for k, v in record.items() if k != "grid"}))
                else:
                    # Legacy providers can supply relief, but cannot assert a datum.
                    dtm = np.ma.asarray(record, dtype=np.float64).filled(np.nan)
            except Exception as exc:
                requested_source_failure = getattr(exc, "provenance", {
                    "status": "UNAVAILABLE", "source": "PUBLIC_ELEVATION_API",
                    "error_type": type(exc).__name__, "support": {"valid_sample_count": 0}})
        elif bounds is not None and is_georeferenced:
            requested_source_failure = {"status": "UNAVAILABLE", "source": "PROVIDER_NOT_INSTALLED",
                                        "support": {"valid_sample_count": 0}}
        if dtm is None:
            dtm = np.broadcast_to(base_srtm_elevation_m + np.linspace(0, terrain_gradient_m, rows)[:, None],
                                  (rows, cols)).copy()
            provenance = {"status": "ASSUMED", "source": "USER_SPECIFIED", "source_surface_type": "ASSUMED_PLANE",
                          "vertical_datum": "ASSUMED_LOCAL_BASE", "datum_verified": False,
                          "bare_earth_certified": False, "independent_of_image": False,
                          "support": {"valid_sample_count": 0},
                          "assumptions": {"base_elevation_m": float(base_srtm_elevation_m),
                                          "terrain_gradient_m": float(terrain_gradient_m)}}
        if dtm.shape != rel_depth.shape:
            raise ValueError("Terrain reference and relative height grid shapes must match.")
        provenance.setdefault("source", "UNVERIFIED_EXTERNAL_GRID")
        provenance.setdefault("status", "UNVERIFIED")
        provenance.setdefault("source_surface_type", "UNKNOWN")
        provenance.setdefault("vertical_datum", "UNKNOWN")
        provenance.setdefault("support", {"valid_sample_count": 0})
        nodata = provenance.get("nodata")
        invalid_terrain = ~np.isfinite(dtm) | (dtm < -500) | (dtm > 10000)
        if nodata is not None:
            invalid_terrain |= dtm == nodata
        dtm = np.where(invalid_terrain, np.nan, dtm).astype(np.float32)
        valid_grid = finite_relative & ~invalid_terrain
        valid_pixels = int(np.sum(valid_grid))
        terrain_status = ("INVALID_SOURCE_SAMPLES" if np.any(invalid_terrain) else
                          "UNAVAILABLE_ASSUMED_PLANE" if requested_source_failure else provenance["status"])
        if requested_source_failure:
            provenance["requested_source_failure"] = requested_source_failure

        structural_heights = (rel_depth * max_structural_height_m).astype(np.float32)
        height_calibrated = False
        irls_metrics = {"status": "NOT_RUN", "converged": False,
                        "reason": "No supported independent AGL controls; terrain is not a structural-height target."}
        if independent_height_controls is not None and use_robust_irls:
            irls_metrics = self._fit_independent_height_controls(rel_depth, independent_height_controls)
            if irls_metrics.get("supported"):
                structural_heights = np.maximum(0.0, irls_metrics["scale"] * rel_depth + irls_metrics["offset"]).astype(np.float32)
                height_calibrated = True
        # Keep the very same source terrain in both decomposition and returned DTM.
        dsm = dtm + structural_heights
        support = provenance.get("support", {})
        terrain_verified = (not np.any(invalid_terrain) and provenance.get("status") in {"AVAILABLE", "VALIDATED"}
                            and provenance.get("independent_of_image") is True
                            and provenance.get("bare_earth_certified") is True
                            and provenance.get("datum_verified") is True
                            and provenance.get("vertical_datum") not in {None, "UNKNOWN", "ASSUMED_LOCAL_BASE"}
                            and isinstance(support, dict) and support.get("valid_sample_count", 0) >= 3)
        horizontal_control = (is_georeferenced and provenance.get("horizontal_crs") not in {None, "UNREFERENCED"}
                              and provenance.get("grid_aligned") is True)
        is_metric = bool(height_calibrated and terrain_verified and horizontal_control and valid_pixels == rel_depth.size)
        vertical_datum = provenance["vertical_datum"] if terrain_verified else (
            "ASSUMED_LOCAL_BASE" if provenance["status"] == "ASSUMED" else "UNKNOWN")

        def finite_stat(array, reducer):
            values = array[np.isfinite(array)]
            return round(float(reducer(values)), 2) if values.size else None

        stats = {
            "status": "SUCCESS" if valid_pixels == rel_depth.size else "NOT_ASSESSED",
            "surface_type": "DSM" if is_metric else "rDSM",
            "surface_semantics": "CALIBRATED_ABSOLUTE_SURFACE" if is_metric else "ASSUMPTION_SCALED_SURFACE",
            "is_metric": is_metric,
            "refusal_reason": None if is_metric else "Metric status withheld: independent height scale and aligned verified bare-earth vertical control are required.",
            "calibration_status": "CALIBRATED" if is_metric else "NOT_ASSESSED" if valid_pixels != rel_depth.size else "UNCALIBRATED",
            "height_scale_status": "INDEPENDENT_AGL_CALIBRATED" if height_calibrated else "ASSUMED",
            "assumed_max_structural_height_m": None if height_calibrated else float(max_structural_height_m),
            "calibration_engine": "INDEPENDENT_AGL_HUBER_IRLS" if height_calibrated else "ASSUMED_HEIGHT_SCALE",
            "terrain_status": terrain_status,
            "terrain_provenance": provenance,
            "terrain_reference_kind": provenance["source_surface_type"],
            "terrain_reference_source": provenance["source"],
            "terrain_reference_authority": "VERIFIED_BARE_EARTH_DTM" if terrain_verified else "UNVERIFIED_TERRAIN_REFERENCE",
            "vertical_datum": vertical_datum,
            "vertical_reference_status": "VERIFIED" if terrain_verified else "UNVERIFIED_OR_ASSUMED",
            "horizontal_reference_status": ("SOURCE_ALIGNED" if horizontal_control else
                                            "SOURCE_COORDINATES_AVAILABLE_ALIGNMENT_UNVERIFIED" if provenance.get("horizontal_crs") else
                                            "BOUNDS_ONLY" if bounds else "UNREFERENCED"),
            "min_elevation_m": finite_stat(dsm, np.min),
            "max_elevation_m": finite_stat(dsm, np.max),
            "mean_elevation_m": finite_stat(dsm, np.mean),
            "base_srtm_m": finite_stat(dtm, np.mean),
            "base_terrain_m": finite_stat(dtm, np.mean),
            "dtm_source": provenance["source"],
            "max_building_height_m": finite_stat(structural_heights, np.max),
            "mean_building_height_m": (finite_stat(structural_heights[structural_heights > 2], np.mean) or 0.0)
                                      if np.all(np.isfinite(structural_heights)) else None,
            "gsd_m": float(gsd_m),
            "ground_sample_dist_m": float(gsd_m),
            "grid_dimensions": [rows, cols],
            "valid_pixel_count": valid_pixels,
            "invalid_pixel_count": int(rel_depth.size - valid_pixels),
            "irls_huber_fit": irls_metrics,
            "inference_provenance": inference_provenance,
            "limitations": ["Appearance is not an observed AGL map.",
                             "A sampled public DSM is not certified bare-earth terrain.",
                             "No neural metric accuracy or safety clearance has been validated."],
        }
        return {"dsm": dsm, "dtm": dtm, "structural_heights": structural_heights, "stats": stats}

    @staticmethod
    def _fit_independent_height_controls(relative: np.ndarray, controls: Dict[str, Any]) -> Dict[str, Any]:
        """Validate independently measured AGL controls before applying a fit."""
        unsupported = {"status": "UNSUPPORTED_CONTROLS", "supported": False, "converged": False}
        if not isinstance(controls, dict):
            return {**unsupported, "reason": "Controls must include observation provenance and pixel coordinates."}
        source = controls.get("source")
        if (controls.get("independent") is not True or controls.get("is_synthetic") is True
                or controls.get("evaluation_only") is True or not isinstance(source, str) or not source.strip()
                or any(word in source.lower() for word in ["synthetic", "ground_truth", "benchmark", "self_fit"])
                or not isinstance(controls.get("measurement_type"), str)
                or controls["measurement_type"] not in {"SURVEYED_AGL", "LIDAR_AGL", "KNOWN_HEIGHT_AGL"}):
            return {**unsupported, "reason": "Independent physical AGL measurements are required; evaluation labels are excluded."}
        try:
            px = np.ma.asarray(controls["pixel_x"], dtype=np.float64).filled(np.nan).reshape(-1)
            py = np.ma.asarray(controls["pixel_y"], dtype=np.float64).filled(np.nan).reshape(-1)
            agl = np.ma.asarray(controls["heights_agl_m"], dtype=np.float64).filled(np.nan).reshape(-1)
            error_limit = float(controls.get("max_rmse_m", 1.5))
            if (px.size < 3 or px.size != py.size or px.size != agl.size
                    or not np.all(np.isfinite(px)) or not np.all(np.isfinite(py))
                    or np.any(px != np.floor(px)) or np.any(py != np.floor(py))
                    or np.any(px < 0) or np.any(px >= relative.shape[1])
                    or np.any(py < 0) or np.any(py >= relative.shape[0])
                    or not np.all(np.isfinite(agl)) or np.any(agl < 0) or np.any(agl > 1000)
                    or not np.isfinite(error_limit) or not 0 < error_limit <= 10):
                raise ValueError("Invalid control coordinates/heights/tolerance.")
            if controls.get("nodata") is not None and np.any(agl == controls["nodata"]):
                raise ValueError("NODATA heights cannot be used as independent observations.")
            if np.ptp(agl) < 0.1:
                raise ValueError("Constant or negligible AGL observations cannot establish a height scale.")
            if np.unique(np.column_stack([px, py]), axis=0).shape[0] < 3:
                raise ValueError("At least three distinct observed pixels are required.")
            x = relative[py.astype(int), px.astype(int)]
            if not np.all(np.isfinite(x)) or np.ptp(x) < 0.1:
                raise ValueError("Control observations do not span a meaningful relative-height range.")
            fit = robust_affine_calibration_irls(x, agl, controls.get("weights"))
            weights = np.ones_like(x) if controls.get("weights") is None else np.ma.asarray(controls["weights"], dtype=np.float64).filled(np.nan).reshape(-1)
            if weights.size != x.size or not np.all(np.isfinite(weights)) or np.any(weights < 0):
                return {**fit, "supported": False, "reason": "Invalid observation weights."}
            supported_pixels = np.column_stack([px[weights > 0], py[weights > 0]])
            distinct_support = int(np.unique(supported_pixels, axis=0).shape[0])
            finite_scene = relative[np.isfinite(relative)]
            fit_range = fit.get("relative_range")
            covered = bool(fit_range and fit_range[0] <= np.min(finite_scene) + 1e-6
                           and fit_range[1] >= np.max(finite_scene) - 1e-6)
            prediction = fit["scale"] * finite_scene + fit["offset"] if fit.get("scale") is not None else None
            supported = bool(fit.get("converged") and distinct_support >= 3 and fit.get("valid_sample_count", 0) >= 3
                             and fit.get("inlier_count", 0) >= 3 and fit.get("rmse_m", np.inf) <= error_limit
                             and covered and prediction is not None and np.all(prediction >= -0.05)
                             and np.all(prediction <= 1000))
            return {**fit, "supported": supported, "source": source, "measurement_type": controls["measurement_type"],
                    "covers_prediction_range": bool(covered),
                    "distinct_supported_pixel_count": distinct_support,
                    "reason": None if supported else "Fit, residual quality, positive height, or prediction-range support failed."}
        except (KeyError, TypeError, ValueError, OverflowError):
            return {**unsupported, "reason": "At least three valid distinct AGL controls with meaningful range are required."}

    def measure_height_at_point(
        self,
        dsm: np.ndarray,
        dtm: np.ndarray,
        pixel_x: int,
        pixel_y: int
    ) -> Dict[str, Any]:
        """
        Query estimated surface/reference heights; this is not laser telemetry.
        """
        rows, cols = dsm.shape
        clamped_x = max(0, min(cols - 1, pixel_x))
        clamped_y = max(0, min(rows - 1, pixel_y))

        abs_elev = float(dsm[clamped_y, clamped_x])
        ground_elev = float(dtm[clamped_y, clamped_x])
        height_agl = max(0.0, abs_elev - ground_elev)

        return {
            "pixel_x": clamped_x,
            "pixel_y": clamped_y,
            "absolute_elevation_m": round(abs_elev, 2),
            "ground_elevation_m": round(ground_elev, 2),
            "height_above_ground_m": round(height_agl, 2)
        }

    def measure_distance_between_points(
        self,
        dsm: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        ground_resolution_m: float = 0.5
    ) -> Dict[str, Any]:
        """
        Calculates 3D Euclidean distance and slope between two arbitrary terrain coordinates.
        """
        rows, cols = dsm.shape
        x1, y1 = max(0, min(cols - 1, p1[0])), max(0, min(rows - 1, p1[1]))
        x2, y2 = max(0, min(cols - 1, p2[0])), max(0, min(rows - 1, p2[1]))

        z1 = float(dsm[y1, x1])
        z2 = float(dsm[y2, x2])

        dx_m = (x2 - x1) * ground_resolution_m
        dy_m = (y2 - y1) * ground_resolution_m
        horizontal_dist_m = float(np.hypot(dx_m, dy_m))
        vertical_diff_m = float(z2 - z1)

        euclidean_3d_dist_m = float(np.sqrt(horizontal_dist_m**2 + vertical_diff_m**2))
        slope_pct = (abs(vertical_diff_m) / max(0.1, horizontal_dist_m)) * 100.0

        return {
            "p1": {"x": x1, "y": y1, "elevation_m": round(z1, 2)},
            "p2": {"x": x2, "y": y2, "elevation_m": round(z2, 2)},
            "horizontal_distance_m": round(horizontal_dist_m, 2),
            "vertical_difference_m": round(vertical_diff_m, 2),
            "elevation_delta_m": round(vertical_diff_m, 2),
            "spatial_3d_distance_m": round(euclidean_3d_dist_m, 2),
            "true_3d_distance_m": round(euclidean_3d_dist_m, 2),
            "slope_percentage": round(slope_pct, 1)
        }

    @staticmethod
    def verify_indian_territory(geo_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Legacy name: report only an approximate WGS84 envelope screen.

        A bounding rectangle cannot verify sovereign territory, SOI approval,
        institutional provenance, or a vertical elevation datum.
        """
        empty = {"is_in_india": None, "status": "UNREFERENCED", "verified": False,
                 "badge": "UNVERIFIED FOOTPRINT", "label": "No usable WGS84 footprint",
                 "zone": "Unknown", "in_india_bounding_envelope": None}
        if not isinstance(geo_metadata, dict):
            return empty
        bounds = geo_metadata.get("bounds_wgs84")
        if bounds is None and geo_metadata.get("crs") == "EPSG:4326":
            bounds = geo_metadata.get("bounds")
        try:
            bounds = np.asarray(bounds, dtype=np.float64)
            if (bounds.shape != (4,) or not np.all(np.isfinite(bounds))
                    or not -180 <= bounds[0] < bounds[2] <= 180
                    or not -90 <= bounds[1] < bounds[3] <= 90):
                return empty
        except (TypeError, ValueError):
            return empty
        center_lon = float((bounds[0] + bounds[2]) / 2)
        center_lat = float((bounds[1] + bounds[3]) / 2)
        in_envelope = bool(68.10 <= bounds[0] and bounds[2] <= 97.42
                           and 6.75 <= bounds[1] and bounds[3] <= 37.10)
        return {**empty, "status": "BOUNDING_ENVELOPE_ONLY", "zone": "WGS84 footprint",
                "in_india_bounding_envelope": in_envelope,
                "label": "Approximate envelope screen; country membership is not verified",
                "center_lat": round(center_lat, 4), "center_lon": round(center_lon, 4)}

    def load_gamus_scene(self, sample_id: str, resample_size: int = 512) -> Dict[str, Any]:
        """Load labelled procedural fixtures or local RGB/raw-AGL sample pairs.

        Local H5 files supply no verified CRS, terrain datum or absolute DSM.
        Their compatibility DSM reference is explicitly fixed base + raw AGL.
        The fixed height prior is never selected from evaluation-label maxima.
        """
        if (isinstance(resample_size, (bool, np.bool_)) or not isinstance(resample_size, (int, np.integer))
                or not 2 <= resample_size <= 2048):
            raise ValueError("resample_size must be an integer between 2 and 2048.")
        if not isinstance(sample_id, str):
            raise ValueError("Scene identifier must be a string.")
        s_norm = sample_id.lower().removeprefix("gamus_").removeprefix("isro_")
        aliases = {"sac": "SAC_AHMEDABAD", "sac_ahmedabad": "SAC_AHMEDABAD",
                   "ahmedabad": "SAC_AHMEDABAD", "isro_sac": "SAC_AHMEDABAD",
                   "synthetic_urban_campus": "SAC_AHMEDABAD", "hilly": "HILLY_RIDGE",
                   "hilly_ridge": "HILLY_RIDGE", "himalaya_01": "HILLY_RIDGE",
                   "synthetic_high_relief_ridge": "HILLY_RIDGE"}
        sample_id = aliases.get(s_norm, s_norm.upper())

        if sample_id.upper() in ["SAC_AHMEDABAD", "AHMEDABAD", "ISRO_SAC"]:
            y_coords = np.linspace(-2.5, 2.5, resample_size)
            x_coords = np.linspace(-2.5, 2.5, resample_size)
            xx, yy = np.meshgrid(x_coords, y_coords)
            base_sac = 52.0
            campus_buildings = np.zeros((resample_size, resample_size), dtype=np.float32)
            campus_buildings[(xx > -1.5) & (xx < -0.5) & (yy > -1.2) & (yy < -0.2)] = 28.0
            campus_buildings[(xx > 0.2) & (xx < 1.4) & (yy > -1.0) & (yy < 0.2)] = 24.0
            campus_buildings[(xx > -0.8) & (xx < 0.0) & (yy > 0.6) & (yy < 1.4)] = 38.0
            campus_buildings[(xx > 0.5) & (xx < 1.6) & (yy > 0.8) & (yy < 1.8)] = 32.0
            sac_elev = base_sac + campus_buildings

            r_c = np.full((resample_size, resample_size), 140, dtype=np.uint8)
            g_c = np.full((resample_size, resample_size), 148, dtype=np.uint8)
            b_c = np.full((resample_size, resample_size), 132, dtype=np.uint8)
            is_bldg = campus_buildings > 5.0
            r_c[is_bldg] = 215
            g_c[is_bldg] = 220
            b_c[is_bldg] = 225
            garden_mask = ((xx**2 + yy**2) < 0.3) | ((xx < -1.8) & (yy > 0.5))
            r_c[garden_mask] = 55
            g_c[garden_mask] = 125
            b_c[garden_mask] = 65
            sac_rgb = np.stack([r_c, g_c, b_c], axis=-1)

            return {
                "scene_id": "synthetic_urban_campus",
                "name": "Synthetic Urban Campus (Procedural Polygonal Benchmark)",
                "terrain_type": "Synthetic Procedural Benchmark",
                "landscape_category": "Urban",
                "is_synthetic": True,
                "rgb_image": sac_rgb,
                "ground_truth_dsm": sac_elev,
                "ground_truth_agl": campus_buildings,
                "ground_truth_dtm": np.full_like(sac_elev, base_sac),
                "reference_surface_semantics": "PROCEDURAL_TERRAIN_PLUS_PROCEDURAL_AGL",
                "reference_provenance": {"source": "PROCEDURAL_POLYGON_FIXTURE", "is_synthetic": True,
                                         "vertical_datum": "ASSUMED_LOCAL_BASE", "evaluation_only": True,
                                         "geography": None},
                "base_elevation_m": base_sac,
                "max_structural_height_m": 25.0,
                "geo_metadata": {
                    "crs": "UNREFERENCED", "bounds": None, "gsd_m": 0.5,
                    "gsd_status": "ASSUMED_FIXTURE_GSD", "vertical_datum": "ASSUMED_LOCAL_BASE"
                }
            }

        if sample_id.upper() in ["HILLY_RIDGE", "HILLY", "HIMALAYA_01"]:
            # Procedural geometry, without any claimed CartoDEM or agency geography.
            rng = np.random.default_rng(42)
            y_coords = np.linspace(-3, 3, resample_size)
            x_coords = np.linspace(-3, 3, resample_size)
            xx, yy = np.meshgrid(x_coords, y_coords)
            ridge_main = 280.0 * np.exp(-((xx * 0.8 - yy * 0.6)**2) / 1.5)
            ridge_spur1 = 120.0 * np.exp(-(((xx + 1.2) * 1.2 + (yy - 0.5) * 1.0)**2) / 1.2)
            ridge_spur2 = 90.0 * np.exp(-(((xx - 1.0) * 1.5 + (yy + 1.2) * 0.8)**2) / 1.0)
            valley_drainage = 45.0 * np.sin(xx * 2.5) * np.cos(yy * 2.0)
            hilly_elev = 1150.0 + ridge_main + ridge_spur1 + ridge_spur2 + valley_drainage

            grad_y, grad_x = np.gradient(hilly_elev, 1.0, 1.0)
            slope = np.sqrt(grad_x**2 + grad_y**2)
            r_band = np.clip(140 + 0.25 * (hilly_elev - 1150) - 0.4 * slope, 40, 220).astype(np.uint8)
            g_band = np.clip(160 - 0.15 * (hilly_elev - 1150) - 0.6 * slope + (valley_drainage * 0.8), 35, 190).astype(np.uint8)
            b_band = np.clip(100 + 0.10 * (hilly_elev - 1150) - 0.3 * slope, 30, 160).astype(np.uint8)
            hilly_rgb = np.stack([r_band, g_band, b_band], axis=-1)
            agl = np.clip(ridge_main * 0.35 + rng.normal(0, 1.5, hilly_elev.shape), 0, 120.0).astype(np.float32)

            return {
                "scene_id": "synthetic_high_relief_ridge",
                "name": "Synthetic High-Relief Mountain Ridge (Procedural Gaussian Benchmark)",
                "terrain_type": "Synthetic Procedural Benchmark",
                "landscape_category": "Hilly",
                "is_synthetic": True,
                "rgb_image": hilly_rgb,
                "ground_truth_dsm": hilly_elev + agl,
                "ground_truth_agl": agl,
                "ground_truth_dtm": hilly_elev,
                "reference_surface_semantics": "PROCEDURAL_TERRAIN_PLUS_PROCEDURAL_AGL",
                "reference_provenance": {"source": "PROCEDURAL_GAUSSIAN_FIXTURE", "is_synthetic": True,
                                         "vertical_datum": "ASSUMED_LOCAL_BASE", "evaluation_only": True,
                                         "geography": None},
                "base_elevation_m": 1150.0,
                "max_structural_height_m": 50.0,
                "geo_metadata": {
                    "crs": "UNREFERENCED", "bounds": None, "gsd_m": 0.6,
                    "gsd_status": "ASSUMED_FIXTURE_GSD", "vertical_datum": "ASSUMED_LOCAL_BASE"
                }
            }
        import h5py

        parts = sample_id.split("_")
        if len(parts) != 3 or parts[0] != "DC" or any(len(p) != 2 or not p.isdigit() for p in parts[1:]):
            raise ValueError(f"Unknown scene identifier: {sample_id}")

        base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "gamus_sample")
        img_path = os.path.join(base_dir, f"{sample_id}_RGB.h5")
        agl_path = os.path.join(base_dir, f"{sample_id}_AGL.h5")

        if not os.path.exists(img_path) or not os.path.exists(agl_path):
            raise FileNotFoundError(f"GAMUS sample {sample_id} not found in {base_dir}")

        with h5py.File(img_path, "r") as f_img, h5py.File(agl_path, "r") as f_agl:
            rgb_data, agl_data = f_img["image"], f_agl["image"]
            if (rgb_data.ndim != 3 or rgb_data.shape[2] != 3 or agl_data.ndim != 2
                    or rgb_data.shape[:2] != agl_data.shape or min(agl_data.shape) < 2):
                raise ValueError("RGB and AGL samples must contain aligned nonempty grids.")
            step = max(1, int(np.ceil(max(agl_data.shape) / resample_size)))
            rgb = np.array(rgb_data[::step, ::step, :])
            agl = np.array(agl_data[::step, ::step])

        base_elev = 15.0
        gt_dsm = base_elev + agl

        scene_names = {
            "DC_02_26": "GAMUS RGB/AGL sample DC_02_26 (Forested)",
            "DC_04_23": "GAMUS RGB/AGL sample DC_04_23 (Urban)",
            "DC_11_33": "GAMUS RGB/AGL sample DC_11_33 (Sparse)"
        }
        landscape_cats = {
            "DC_02_26": "Forested",
            "DC_04_23": "Urban",
            "DC_11_33": "Sparse"
        }

        # These local H5 files contain no georeferencing attributes.
        geo_metadata = {
            "crs": "UNREFERENCED", "bounds": None,
            "gsd_m": float(0.5 * step), "gsd_status": "ASSUMED_0_5M_NATIVE_GSD_WITH_DECIMATION",
            "pixel_stride": step, "vertical_datum": "ASSUMED_LOCAL_BASE"
        }

        return {
            "scene_id": f"gamus_{sample_id.lower()}",
            "name": scene_names.get(sample_id, f"GAMUS RGB/AGL sample {sample_id}"),
            "terrain_type": "Paired RGB/raw AGL sample (no absolute DSM control)",
            "landscape_category": landscape_cats.get(sample_id, "Urban"),
            "is_synthetic": False,
            "rgb_image": rgb,
            "ground_truth_dsm": gt_dsm,
            "ground_truth_agl": agl,
            "reference_surface_semantics": "FIXED_BASE_PLUS_RAW_AGL_NOT_OBSERVED_ABSOLUTE_DSM",
            "reference_provenance": {"source": "LOCAL_GAMUS_RGB_AGL_H5", "is_synthetic": False,
                                     "raw_agl_preserved": True, "evaluation_only": True,
                                     "absolute_dsm_available": False, "terrain_reference_available": False,
                                     "fixed_base_elevation_m": base_elev, "vertical_datum": "ASSUMED_LOCAL_BASE",
                                     "resampling": "STRIDED_RAW_VALUES", "pixel_stride": step,
                                     "limitations": ["No verified CRS, GSD or vertical datum in the local H5 files.",
                                                     "Negative/nonfinite AGL labels are retained, not clipped or used for calibration."]},
            "base_elevation_m": base_elev,
            "max_structural_height_m": 25.0, # Autonomous blind prior (zero label leakage from evaluation AGL)
            "geo_metadata": geo_metadata
        }

    def process_image_file(
        self,
        file_bytes: bytes,
        filename: str,
        is_georeferenced: bool = False,
        base_srtm_elevation_m: float = 50.0,
        max_structural_height_m: float = 45.0,
        target_resample_size: int = 512
    ) -> Dict[str, Any]:
        """
        Ingests user-uploaded satellite imagery (PNG, JPG, TIFF, GeoTIFF).
        Preserves declared horizontal coordinates and resampled pixel spacing.
        Declared CRS does not supply a calibrated height or vertical datum.
        """
        if (isinstance(target_resample_size, (bool, np.bool_))
                or not isinstance(target_resample_size, (int, np.integer))
                or not 2 <= target_resample_size <= 2048):
            raise ValueError("target_resample_size must be an integer between 2 and 2048.")
        if not isinstance(file_bytes, bytes) or not 0 < len(file_bytes) <= 50 * 1024 * 1024:
            raise ValueError("Input must contain between 1 byte and 50 MiB of image data.")
        if not isinstance(filename, str) or not filename:
            raise ValueError("An image filename is required.")
        requested_georeferenced = bool(is_georeferenced)
        is_georeferenced = False
        geo_meta = {
            "crs": "UNREFERENCED",
            "bounds": None,
            "bounds_wgs84": None,
            "transform": None,
            "gsd_m": 0.6,
            "gsd_status": "ASSUMED_DEFAULT",
            "vertical_datum": "UNKNOWN",
            "requested_georeferenced": requested_georeferenced,
            "horizontal_reference_status": "UNREFERENCED"
        }

        is_tiff = filename.lower().endswith((".tif", ".tiff"))
        rgb_image = None

        if is_tiff and HAS_RASTERIO:
            try:
                with rasterio.open(io.BytesIO(file_bytes)) as src:
                    if src.width * src.height > 64 * 1024 * 1024 or min(src.width, src.height) < 2:
                        raise ValueError("Source image must contain between 2x2 pixels and 64 megapixels.")
                    scale = min(1.0, target_resample_size / max(src.width, src.height))
                    new_w, new_h = max(2, int(src.width * scale)), max(2, int(src.height * scale))
                    bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
                    raster = src.read(bands, out_shape=(3, new_h, new_w), masked=True,
                                      resampling=rasterio.enums.Resampling.bilinear)
                    if np.any(np.ma.getmaskarray(raster)):
                        raise ValueError("Image contains NODATA pixels; no supported relative surface can be extracted.")
                    rgb_image = np.moveaxis(np.asarray(raster), 0, -1)
                    transform = src.transform * rasterio.Affine.scale(src.width / new_w, src.height / new_h)
                    geo_meta["ingestion_status"] = "RASTERIO_READ"
                    geo_meta["source_grid_dimensions"] = [src.height, src.width]
                    geo_meta["source_transform"] = [float(v) for v in list(src.transform)[:6]]
                    if src.crs:
                        geo_meta["crs"] = str(src.crs)
                        is_georeferenced = True
                        geo_meta["horizontal_reference_status"] = "RASTER_DECLARED_CRS"
                        geo_meta["transform"] = [float(v) for v in list(transform)[:6]]
                        geo_meta["bounds"] = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
                        from rasterio.warp import transform_bounds
                        geo_meta["bounds_wgs84"] = list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))
                        if src.crs.is_projected:
                            _, factor = src.crs.linear_units_factor
                            gsd_x = np.hypot(transform.a, transform.d) * factor
                            gsd_y = np.hypot(transform.b, transform.e) * factor
                            geo_meta["gsd_status"] = "PROJECTED_CRS_LINEAR_UNITS"
                        elif src.crs.is_geographic:
                            center_lat = (geo_meta["bounds_wgs84"][1] + geo_meta["bounds_wgs84"][3]) / 2
                            mx, my = 111320 * np.cos(np.radians(center_lat)), 111132
                            # Geographic pixel sizes are angular; convert locally
                            # and label the approximation rather than use degrees as metres.
                            gsd_x = np.hypot(transform.a * mx, transform.d * my)
                            gsd_y = np.hypot(transform.b * mx, transform.e * my)
                            geo_meta["gsd_status"] = "APPROXIMATE_LOCAL_GEOGRAPHIC_CONVERSION"
                        else:
                            raise ValueError("CRS cannot provide usable horizontal sampling units.")
                        geo_meta["gsd_m"] = float((gsd_x + gsd_y) / 2)
                        if not np.isfinite(geo_meta["gsd_m"]) or geo_meta["gsd_m"] <= 0:
                            raise ValueError("Raster transform does not provide positive pixel spacing.")
            except Exception as e:
                raise ValueError(f"TIFF ingestion failed: {type(e).__name__}: {e}") from e

        if rgb_image is None:
            with Image.open(io.BytesIO(file_bytes)) as pil_img:
                if pil_img.width * pil_img.height > 64 * 1024 * 1024 or min(pil_img.size) < 2:
                    raise ValueError("Source image must contain between 2x2 pixels and 64 megapixels.")
                pil_img = pil_img.convert("RGB")
                pil_img.thumbnail((target_resample_size, target_resample_size), Image.Resampling.LANCZOS)
                rgb_image = np.array(pil_img)
            geo_meta["ingestion_status"] = "TIFF_PIL_NO_GEORASTER_READER" if is_tiff else "PIL_OPTICAL_IMAGE"

        # Bound any remaining image and disclose radiometric normalization.
        h, w = rgb_image.shape[:2]
        if max(h, w) > target_resample_size:
            scale = target_resample_size / max(h, w)
            new_w, new_h = max(2, int(w * scale)), max(2, int(h * scale))
            rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if not np.all(np.isfinite(rgb_image)):
            raise ValueError("Image contains nonfinite radiometry.")
        if rgb_image.dtype != np.uint8:
            low, high = np.percentile(rgb_image, [2, 98])
            rgb_image = np.clip((rgb_image.astype(np.float64) - low) / max(high - low, 1e-8) * 255, 0, 255).astype(np.uint8)
            geo_meta["radiometry_status"] = "PERCENTILE_SCALED_FOR_APPEARANCE_HEURISTICS"

        # 1. Monocular Relative Depth Extraction
        rel_depth = self.extract_relative_depth(rgb_image)

        # 2. Metric Scale Calibration (with real SRTM when georeferenced)
        calib = self.calibrate_to_absolute_dsm(
            rel_depth=rel_depth,
            base_srtm_elevation_m=base_srtm_elevation_m if is_georeferenced else 0.0,
            max_structural_height_m=max_structural_height_m,
            gsd_m=geo_meta.get("gsd_m", 0.6),
            geo_bounds=geo_meta.get("bounds_wgs84") if is_georeferenced else None,
            is_georeferenced=is_georeferenced
        )
        geo_meta["vertical_datum"] = calib["stats"]["vertical_datum"]
        geo_meta["height_calibration_status"] = calib["stats"]["calibration_status"]

        return {
            "scene_id": "uploaded_" + Path(filename).stem,
            "name": f"Uploaded: {filename}",
            "terrain_type": "Georeferenced optical image (uncalibrated heights)" if is_georeferenced else "Standard optical image (uncalibrated heights)",
            "model_mode": "ABSOLUTE_METRIC_DSM" if calib["stats"]["is_metric"] else "ASSUMED_SCALE_RDSM",
            "is_georeferenced": is_georeferenced,
            "is_metric": calib["stats"]["is_metric"],
            "inference_provenance": self.last_inference_provenance,
            "rgb_image": rgb_image,
            "dsm": calib["dsm"],
            "dtm": calib["dtm"],
            "structural_heights": calib["structural_heights"],
            "stats": calib["stats"],
            "geo_metadata": geo_meta
        }

    def export_dsm_geotiff(
        self,
        dsm: np.ndarray,
        geo_meta: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Export float32 values with declared horizontal metadata and scale tags.
        Unreferenced TIFFs remain unreferenced; no normalized integer raster
        silently replaces elevation values or a failed georeferenced export.
        """
        arrays, _, _, error = self._validate_safety_grids(1.0, surface=dsm)
        if error:
            raise ValueError(error)
        surface = arrays["surface"].astype(np.float32)
        rows, cols = surface.shape
        geo_meta = geo_meta or {}
        georeferenced = geo_meta.get("crs") not in {None, "UNREFERENCED"} and geo_meta.get("bounds") is not None
        if georeferenced and not HAS_RASTERIO:
            raise RuntimeError("Georeferenced float32 export requires rasterio; no geospatial substitute is available.")
        if HAS_RASTERIO:
            options = {}
            if georeferenced:
                bounds = np.asarray(geo_meta["bounds"], dtype=np.float64)
                if bounds.shape != (4,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
                    raise ValueError("Export bounds must be finite and ordered in the declared CRS.")
                if geo_meta.get("transform") is not None:
                    affine = np.asarray(geo_meta["transform"], dtype=np.float64)
                    if affine.shape != (6,) or not np.all(np.isfinite(affine)):
                        raise ValueError("Export transform must contain six finite affine coefficients.")
                    transform = rasterio.Affine(*affine)
                    if abs(transform.a * transform.e - transform.b * transform.d) < 1e-20:
                        raise ValueError("Export transform is degenerate.")
                else:
                    transform = from_bounds(*bounds, cols, rows)
                options = {"crs": geo_meta["crs"], "transform": transform}
            memfile = io.BytesIO()
            with rasterio.open(memfile, "w", driver="GTiff", height=rows, width=cols, count=1,
                               dtype="float32", nodata=-9999.0, **options) as dst:
                dst.write(surface, 1)
                dst.update_tags(VERTICAL_DATUM=str(geo_meta.get("vertical_datum", "UNKNOWN")),
                                HEIGHT_CALIBRATION_STATUS=str(geo_meta.get("height_calibration_status", "UNSPECIFIED")),
                                VERTICAL_ACCURACY="NOT_ESTABLISHED")
            return memfile.getvalue()
        pil_tiff = Image.fromarray(surface)
        buffer = io.BytesIO()
        pil_tiff.save(buffer, format="TIFF")
        return buffer.getvalue()

    def export_dsm_tiff(self, dsm: np.ndarray) -> bytes:
        return self.export_dsm_geotiff(dsm)


    def export_dsm_obj(self, dsm: np.ndarray, step: int = 4, vertical_exag: float = 1.0) -> bytes:
        """
        Exports active DSM as standard Wavefront OBJ 3D mesh for Blender, Unity, and GIS software.
        """
        arrays, _, _, error = self._validate_safety_grids(1.0, surface=dsm)
        if error:
            raise ValueError(error)
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)) or not 1 <= step <= 2048:
            raise ValueError("OBJ sampling step must be an integer between 1 and 2048.")
        if not np.isfinite(vertical_exag) or not 0 < vertical_exag <= 100:
            raise ValueError("Vertical display exaggeration must be finite, positive and no greater than 100.")
        dsm = arrays["surface"]
        h, w = dsm.shape
        sub_dsm = dsm[::step, ::step]
        sub_h, sub_w = sub_dsm.shape
        
        base_z = float(np.min(sub_dsm))
        
        lines = [
            "# DepthWizard (SIH26175) visualization mesh; pixel-grid coordinates\n",
            "# Vertical values depend on source scale/datum; no metric accuracy is certified\n",
            f"# Resolution: {sub_w}x{sub_h} vertices\n"
        ]
        
        # Vertices & UV coordinates
        for y in range(sub_h):
            for x in range(sub_w):
                z = (float(sub_dsm[y, x]) - base_z) * vertical_exag
                lines.append(f"v {x} {y} {z:.2f}\n")
                lines.append(f"vt {x/max(sub_w-1, 1):.4f} {1.0 - y/max(sub_h-1, 1):.4f}\n")
                
        # Faces (quad split into two triangles)
        for y in range(sub_h - 1):
            for x in range(sub_w - 1):
                i1 = y * sub_w + x + 1
                i2 = y * sub_w + (x + 1) + 1
                i3 = (y + 1) * sub_w + (x + 1) + 1
                i4 = (y + 1) * sub_w + x + 1
                lines.append(f"f {i1}/{i1} {i2}/{i2} {i3}/{i3}\n")
                lines.append(f"f {i1}/{i1} {i3}/{i3} {i4}/{i4}\n")
                
        return "".join(lines).encode("utf-8")


    @staticmethod
    def _validate_safety_grids(ground_res_m, **inputs):
        """Fail closed on unsupported sizes, sampling, shapes and missing data."""
        diagnostics = {"total_pixel_count": 0, "valid_pixel_count": 0,
                       "invalid_pixel_count": 0, "grid_dimensions": None}
        arrays = {}
        try:
            if isinstance(ground_res_m, (bool, np.bool_)):
                raise ValueError("Ground sampling distance must be numeric, not boolean.")
            resolution = float(ground_res_m)
            if not np.isfinite(resolution) or not 1e-6 <= resolution <= 100000:
                raise ValueError("Ground sampling distance must be finite, positive and bounded.")
            shape = None
            for name, value in inputs.items():
                # Masked pixels are unavailable observations even if their hidden
                # backing values happen to be plausible and finite.
                grid = np.ma.asarray(value, dtype=np.float64).filled(np.nan)
                if grid.ndim != 2 or min(grid.shape) < 2 or max(grid.shape) > 2048:
                    raise ValueError(f"{name} must be a 2D grid with sides between 2 and 2048.")
                if shape is None:
                    shape = grid.shape
                    diagnostics.update(total_pixel_count=int(grid.size), grid_dimensions=list(shape))
                elif grid.shape != shape:
                    raise ValueError("All surface grids must have identical shapes.")
                arrays[name] = grid
            valid = np.ones(shape, dtype=bool)
            for grid in arrays.values():
                valid &= np.isfinite(grid) & (np.abs(grid) <= 1e6) & (grid != -9999) & (grid != -32768)
            diagnostics.update(valid_pixel_count=int(np.sum(valid)), invalid_pixel_count=int(np.sum(~valid)))
            if not np.all(valid):
                raise ValueError("Nonfinite or NODATA pixels prevent complete local-footprint assessment.")
            return arrays, resolution, diagnostics, None
        except (TypeError, ValueError, OverflowError) as exc:
            return arrays, None, diagnostics, str(exc)

    def simulate_flood(
        self,
        dsm: np.ndarray,
        water_level_m: float,
        structural_heights: Optional[np.ndarray] = None,
        ground_res_m: float = 0.5
    ) -> Dict[str, Any]:
        """Static water-level intersection; not a hydrological forecast."""
        inputs = {"dsm": dsm}
        if structural_heights is not None:
            inputs["structural_heights"] = structural_heights
        arrays, ground_res_m, diagnostics, error = self._validate_safety_grids(ground_res_m, **inputs)
        try:
            water_level_m = float(water_level_m)
            if not np.isfinite(water_level_m) or not -500 <= water_level_m <= 10000:
                raise ValueError("Water level must be a finite plausible surface-coordinate level.")
        except (TypeError, ValueError, OverflowError) as exc:
            error = str(exc)
        if structural_heights is not None and "structural_heights" in arrays and np.any(arrays["structural_heights"] < 0):
            error = "Structural heights must be nonnegative."
        if error:
            return {"status": "NOT_ASSESSED", "reason": error, "diagnostics": diagnostics,
                    "assessment_scope": "STATIC_WATER_LEVEL_GEOMETRY_ONLY", "water_level_m": None,
                    "inundated_area_m2": None, "inundated_hectares": None, "submergence_pct": None,
                    "max_depth_m": None, "mean_depth_m": None, "water_volume_m3": None,
                    "affected_structures_count": 0}
        dsm = arrays["dsm"]
        structural_heights = arrays.get("structural_heights")
        cell_area_m2 = ground_res_m * ground_res_m
        inundated_mask = dsm <= water_level_m
        inundated_cells = int(np.sum(inundated_mask))
        total_cells = int(dsm.size)
        
        inundated_area_m2 = round(inundated_cells * cell_area_m2, 2)
        inundated_hectares = round(inundated_area_m2 / 10000.0, 4)
        submergence_pct = round((inundated_cells / max(total_cells, 1)) * 100.0, 2)
        
        water_depths = np.maximum(water_level_m - dsm, 0.0)
        max_depth = round(float(np.max(water_depths)), 2) if inundated_cells > 0 else 0.0
        mean_depth = round(float(np.mean(water_depths[inundated_mask])), 2) if inundated_cells > 0 else 0.0
        water_volume_m3 = round(float(np.sum(water_depths[inundated_mask]) * cell_area_m2), 1) if inundated_cells > 0 else 0.0
        
        affected_structures = 0
        if structural_heights is not None:
            struct_submerged = (structural_heights > 1.5) & (water_depths > 0.3)
            labeled, num_features = ndimage.label(struct_submerged)
            affected_structures = int(num_features)
            
        return {
            "status": "SUCCESS",
            "assessment_scope": "STATIC_WATER_LEVEL_GEOMETRY_ONLY",
            "diagnostics": diagnostics,
            "limitations": ["No flow routing, drainage, rainfall, flood defences or hydraulic model is assessed.",
                             "Results depend on valid surface scale, vertical datum and water-level assumptions."],
            "water_level_m": round(water_level_m, 2),
            "inundated_area_m2": inundated_area_m2,
            "inundated_hectares": inundated_hectares,
            "submergence_pct": submergence_pct,
            "max_depth_m": max_depth,
            "mean_depth_m": mean_depth,
            "water_volume_m3": water_volume_m3,
            "affected_structures_count": affected_structures
        }

    def detect_landing_zones(
        self,
        dsm: np.ndarray,
        dtm: np.ndarray,
        structural_heights: np.ndarray,
        ground_res_m: float = 0.5,
        pad_radius_m: float = 8.0,
        max_slope_deg: float = 5.0
    ) -> Dict[str, Any]:
        """Screen complete local disks for geometric slope/obstacle criteria.

        Candidate centres are actual valid clearance pixels, not component
        centroids. Pixel extent and scene borders reduce reported clear radius.
        Unseen obstacles, bearing capacity and aviation criteria are unassessed.
        """
        arrays, ground_res_m, diagnostics, error = self._validate_safety_grids(
            ground_res_m, dsm=dsm, dtm=dtm, structural_heights=structural_heights)
        try:
            pad_radius_m, max_slope_deg = float(pad_radius_m), float(max_slope_deg)
            if not np.isfinite(pad_radius_m) or not 0 < pad_radius_m <= 500:
                raise ValueError("Pad radius must be finite and between 0 and 500 metres.")
            if not np.isfinite(max_slope_deg) or not 0 < max_slope_deg <= 45:
                raise ValueError("Slope threshold must be finite and between 0 and 45 degrees.")
        except (TypeError, ValueError, OverflowError) as exc:
            error = str(exc)
        diagnostics.update(clear_pixel_count=0, candidate_center_count=0, candidate_component_count=0)
        result = {"status": "NOT_ASSESSED" if error else "SUCCESS", "assessment_scope": "GEOMETRIC_SCREEN_ONLY",
                  "detected_zones_count": 0, "candidate_zones": [], "diagnostics": diagnostics,
                  "pad_radius_m": pad_radius_m if not error else None,
                  "max_slope_limit_deg": max_slope_deg if not error else None,
                  "flight_safety_assessed": False,
                  "missing_factors": ["Unobserved obstacles and approach/departure paths", "Surface bearing capacity",
                                      "Aircraft performance, weather and operational reconnaissance"],
                  "reason": error}
        if error:
            return result
        dsm, dtm, structural_heights = arrays["dsm"], arrays["dtm"], arrays["structural_heights"]
        if (np.any(structural_heights < 0)
                or not np.allclose(dsm, dtm + structural_heights, rtol=1e-6, atol=0.05)):
            return {**result, "status": "NOT_ASSESSED", "reason": "DSM/terrain/structural-height decomposition is inconsistent."}
        h, w = dtm.shape
        pad_px = int(np.ceil(pad_radius_m / ground_res_m + 1 / np.sqrt(2)))
        if 2 * pad_px + 1 > min(h, w):
            return {**result, "status": "NOT_ASSESSED", "reason": "Requested complete footprint is larger than the available scene."}
        gy, gx = np.gradient(dtm, ground_res_m, ground_res_m)
        slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
        clear_cells = (slope_deg <= max_slope_deg) & (structural_heights <= 0.5)
        # EDT distances are centre-to-centre. Subtract half a pixel diagonal
        # so no observed obstacle pixel can intrude into the proposed disk.
        distance = ndimage.distance_transform_edt(np.pad(clear_cells, 1, constant_values=False))[1:-1, 1:-1]
        radius = np.maximum(0.0, distance * ground_res_m - ground_res_m / np.sqrt(2))
        centers = clear_cells & (radius >= pad_radius_m)
        labeled, num_features = ndimage.label(centers)
        result["diagnostics"].update(clear_pixel_count=int(np.sum(clear_cells)),
                                     candidate_center_count=int(np.sum(centers)),
                                     candidate_component_count=int(num_features))
        if num_features == 0:
            result["reason"] = "No complete local footprint meets the geometric thresholds."
            return result
        yy, xx = np.ogrid[-pad_px:pad_px + 1, -pad_px:pad_px + 1]
        disk = np.hypot(xx, yy) * ground_res_m <= pad_radius_m + ground_res_m / np.sqrt(2)
        zones = []
        for component in range(1, min(num_features, 14) + 1):
            mask = labeled == component
            iy, ix = np.unravel_index(np.argmax(np.where(mask, radius, -1.0)), mask.shape)
            if (not centers[iy, ix] or iy < pad_px or ix < pad_px
                    or iy + pad_px >= h or ix + pad_px >= w):
                continue
            region = np.s_[iy - pad_px:iy + pad_px + 1, ix - pad_px:ix + pad_px + 1]
            if not np.all(clear_cells[region][disk]):
                continue
            local_slope = float(np.max(slope_deg[region][disk]))
            local_relief = float(np.ptp(dsm[region][disk]))
            relief_limit = 2 * (pad_radius_m + ground_res_m / np.sqrt(2)) * np.tan(np.radians(max_slope_deg)) + 0.5
            if local_relief > relief_limit:
                continue
            zones.append({"zone_id": f"GEOM-{component:02d}", "pixel_x": int(ix), "pixel_y": int(iy),
                          "norm_x": round(int(ix) / (w - 1), 4), "norm_y": round(int(iy) / (h - 1), 4),
                          "elevation_m": round(float(dsm[iy, ix]), 2), "slope_deg": round(local_slope, 2),
                          "local_relief_m": round(local_relief, 3), "footprint_radius_m": pad_radius_m,
                          "clearance_diameter_m": float(np.floor(radius[iy, ix] * 20) / 10),
                          "suitability": "GEOMETRIC_CANDIDATE_ONLY", "flight_safety_assessed": False})
        result.update(detected_zones_count=len(zones), candidate_zones=zones)
        if not zones:
            result["reason"] = "No complete local footprint meets the geometric thresholds."
        return result

    def screen_landslide_risk(
        self,
        dtm: np.ndarray,
        ground_res_m: float = 0.5
    ) -> Dict[str, Any]:
        """Slope-band and relative-relief geometry, not landslide probability.

        A low-slope pixel is not established as stable. No BIS/TEHD rating can
        be derived from these two geometric quantities alone. Legacy numeric
        hazard keys refer only to slope-band area for caller compatibility.
        """
        arrays, ground_res_m, diagnostics, error = self._validate_safety_grids(ground_res_m, dtm=dtm)
        result = {
            "status": "NOT_ASSESSED" if error else "SUCCESS",
            "assessment_scope": "GEOMETRIC_SCREEN_ONLY", "reason": error, "diagnostics": diagnostics,
            "landslide_risk_assessed": False, "geotechnical_assessment_status": "NOT_ASSESSED",
            "standards_compliance_assessed": False,
            "missing_geotechnical_factors": ["Lithology and discontinuities", "Soil strength and weathering",
                                             "Groundwater and drainage", "Rainfall and seismic loading",
                                             "Land cover, excavation and field investigation"],
            "metrics_semantics": "Legacy hazard keys represent slope-band areas only, not hazard or stability.",
            "critical_hazard_pct": None, "moderate_hazard_pct": None, "critical_hazard_area_m2": None,
            "low_slope_pct": None, "mean_slope_deg": None, "max_slope_deg": None, "relative_relief_m": None,
            "slope_band_counts": {},
        }
        if error:
            return result
        dtm = arrays["dtm"]
        gy, gx = np.gradient(dtm, ground_res_m, ground_res_m)
        slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
        low = int(np.sum(slope_deg < 15))
        middle = int(np.sum((slope_deg >= 15) & (slope_deg < 30)))
        high = int(np.sum(slope_deg >= 30))
        result.update(
            critical_hazard_pct=round(high / dtm.size * 100, 2),
            moderate_hazard_pct=round(middle / dtm.size * 100, 2),
            low_slope_pct=round(low / dtm.size * 100, 2),
            critical_hazard_area_m2=round(high * ground_res_m ** 2, 2),
            mean_slope_deg=round(float(np.mean(slope_deg)), 2),
            max_slope_deg=round(float(np.max(slope_deg)), 2),
            relative_relief_m=round(float(np.ptp(dtm)), 3),
            slope_band_counts={"below_15_deg": low, "15_to_30_deg": middle, "at_least_30_deg": high})
        return result
