"""Descriptive elevation errors, with explicit support and source provenance.

These statistics are measurements of array differences, not agency approval,
operational clearance, or a certification rubric.
"""

import numpy as np
from typing import Dict, Any


class DepthWizardBenchmark:
    METRIC_KEYS = (
        "rmse_meters", "mae_meters", "bias_meters", "nmad_meters",
        "std_error_m", "pearson_correlation_r", "r_squared", "le90_meters",
        "max_residual_error_m",
    )
    SLOPE_KEYS = (
        "flat_terrain_below_5deg_rmse_m", "moderate_slopes_5_to_15deg_rmse_m",
        "steep_terrain_above_15deg_rmse_m",
    )
    TERRAIN_SPECS = (
        {"id": "isro_sac_ahmedabad", "cat": "Urban", "label": "Urban (procedural campus fixture)"},
        {"id": "gamus_dc_11_33", "cat": "Sparse", "label": "Sparse (GAMUS DC_11_33)"},
        {"id": "gamus_hilly_ridge", "cat": "Hilly", "label": "Hilly (procedural ridge fixture)"},
        {"id": "gamus_dc_02_26", "cat": "Forested", "label": "Forested (GAMUS DC_02_26)"},
    )

    @staticmethod
    def _number(value, digits=None):
        if value is None:
            return None
        value = float(value)
        return (round(value, digits) if digits is not None else value) if np.isfinite(value) else None

    @classmethod
    def _json_safe(cls, value):
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return cls._json_safe(value.tolist())
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return cls._json_safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    @classmethod
    def _unassessed(cls, terrain_type, reason, total=0, valid=0):
        return {
            "terrain_type": terrain_type, "status": "NOT_ASSESSED",
            "reason": reason, "certification": "NOT CERTIFIED",
            **{key: None for key in cls.METRIC_KEYS},
            "error_percentiles": {"50th_percentile_m": None, "le90_m": None, "95th_percentile_m": None},
            "slope_stratification": {key: None for key in cls.SLOPE_KEYS},
            "slope_stratum_sample_counts": {key: 0 for key in cls.SLOPE_KEYS},
            "slope_stratum_status": {key: "NOT_ASSESSED" for key in cls.SLOPE_KEYS},
            "slope_assessment_status": "NOT_ASSESSED",
            "sample_points_evaluated": valid, "valid_sample_count": valid,
            "total_sample_count": total, "invalid_sample_count": total - valid,
            "sufficient_statistics": None,
        }

    @classmethod
    def evaluate(
        cls, predicted_dsm: np.ndarray, ground_truth_dsm: np.ndarray,
        terrain_type: str = "Urban", gsd_m: float = 0.6,
        valid_mask=None, nodata_value=-9999.0,
    ) -> Dict[str, Any]:
        """Evaluate only jointly finite, unmasked, non-nodata reference support.

        Constant-array correlations and empty slope strata are undefined (null).
        A partial-support result is not presented as a complete scene evaluation.
        """
        try:
            pred_ma = np.ma.asarray(predicted_dsm, dtype=np.float64)
            gt_ma = np.ma.asarray(ground_truth_dsm, dtype=np.float64)
        except (TypeError, ValueError):
            return cls._unassessed(terrain_type, "INVALID_ARRAY_INPUT")
        if pred_ma.shape != gt_ma.shape or pred_ma.ndim != 2:
            return cls._unassessed(terrain_type, "SHAPE_MISMATCH_OR_NON_GRID_INPUT")
        total = int(pred_ma.size)
        pred = np.ma.getdata(pred_ma)
        gt = np.ma.getdata(gt_ma)
        mask = np.isfinite(pred) & np.isfinite(gt)
        mask &= ~np.ma.getmaskarray(pred_ma) & ~np.ma.getmaskarray(gt_ma)
        if nodata_value is not None:
            for nodata in np.atleast_1d(nodata_value):
                mask &= (pred != nodata) & (gt != nodata)
            mask &= (pred != -32768) & (gt != -32768)
        if valid_mask is not None:
            support = np.ma.asarray(valid_mask)
            if support.shape != mask.shape:
                return cls._unassessed(terrain_type, "SUPPORT_MASK_SHAPE_MISMATCH", total)
            try:
                finite_support = np.isfinite(np.ma.getdata(support))
            except TypeError:
                return cls._unassessed(terrain_type, "INVALID_SUPPORT_MASK", total)
            mask &= finite_support & np.asarray(np.ma.getdata(support), dtype=bool) & ~np.ma.getmaskarray(support)
        with np.errstate(over="ignore", invalid="ignore"):
            residual_grid = pred - gt
        mask &= np.isfinite(residual_grid)
        count = int(np.count_nonzero(mask))
        if not count:
            return cls._unassessed(terrain_type, "NO_VALID_REFERENCE_SUPPORT", total)

        residuals, p, g = residual_grid[mask], pred[mask], gt[mask]
        absolute = np.abs(residuals)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            squared_sum = float(np.sum(residuals ** 2))
            rmse = float(np.sqrt(squared_sum / count))
            mae = float(np.mean(absolute))
            bias = float(np.mean(residuals))
            pred_mean, gt_mean = float(np.mean(p)), float(np.mean(g))
            pd, gd = p - pred_mean, g - gt_mean
            pred_m2, gt_m2 = float(np.sum(pd ** 2)), float(np.sum(gd ** 2))
            covariance = float(np.sum(pd * gd))
        if not all(np.isfinite(v) for v in (squared_sum, rmse, mae, bias, pred_m2, gt_m2, covariance)):
            return cls._unassessed(terrain_type, "NUMERICAL_OVERFLOW", total, count)
        pearson = None
        if count >= 2 and pred_m2 > 0 and gt_m2 > 0:
            pearson = float(np.clip((covariance / np.sqrt(pred_m2)) / np.sqrt(gt_m2), -1, 1))
        r2 = 1.0 - squared_sum / gt_m2 if gt_m2 > 0 else None
        median = float(np.median(residuals))
        nmad = float(1.4826 * np.median(np.abs(residuals - median)))
        p50, p90, p95 = (float(v) for v in np.percentile(absolute, [50, 90, 95]))

        strata = {key: None for key in cls.SLOPE_KEYS}
        stratum_counts = {key: 0 for key in cls.SLOPE_KEYS}
        try:
            spacing = float(gsd_m)
            slope_assessable = np.isfinite(spacing) and spacing > 0 and min(pred.shape, default=0) >= 2
        except (TypeError, ValueError):
            slope_assessable = False
        if slope_assessable:
            slope_reference = np.where(mask, gt, np.nan)
            with np.errstate(invalid="ignore", over="ignore"):
                dy, dx = np.gradient(slope_reference, spacing, spacing)
                slope = np.degrees(np.arctan(np.hypot(dx, dy)))
            slope_support = mask & np.isfinite(slope)
            partitions = (slope < 5, (slope >= 5) & (slope <= 15), slope > 15)
            for key, partition in zip(cls.SLOPE_KEYS, partitions):
                support = slope_support & partition
                stratum_counts[key] = int(np.count_nonzero(support))
                if stratum_counts[key]:
                    strata[key] = cls._number(np.sqrt(np.mean(residual_grid[support] ** 2)), 2)

        return {
            "terrain_type": terrain_type, "status": "ASSESSED" if count == total else "PARTIAL",
            "reason": None if count == total else "INVALID_OR_MASKED_SAMPLES_EXCLUDED",
            "certification": "NOT CERTIFIED",
            "rmse_meters": cls._number(rmse, 2), "mae_meters": cls._number(mae, 2),
            "bias_meters": cls._number(bias, 2), "nmad_meters": cls._number(nmad, 2),
            "std_error_m": cls._number(np.std(residuals), 2),
            "pearson_correlation_r": cls._number(pearson, 4), "r_squared": cls._number(r2, 4),
            "le90_meters": cls._number(p90, 2), "max_residual_error_m": cls._number(np.max(absolute), 2),
            "error_percentiles": {"50th_percentile_m": cls._number(p50, 2), "le90_m": cls._number(p90, 2), "95th_percentile_m": cls._number(p95, 2)},
            "slope_stratification": strata, "slope_stratum_sample_counts": stratum_counts,
            "slope_assessment_status": "ASSESSED" if any(stratum_counts.values()) else "NOT_ASSESSED",
            "slope_stratum_status": {key: "ASSESSED" if value else "NOT_ASSESSED" for key, value in stratum_counts.items()},
            "sample_points_evaluated": count, "valid_sample_count": count,
            "total_sample_count": total, "invalid_sample_count": total - count,
            "sufficient_statistics": {
                "count": count, "squared_error_sum": squared_sum,
                "absolute_error_sum": float(np.sum(absolute)), "error_sum": float(np.sum(residuals)),
                "predicted_mean": pred_mean, "reference_mean": gt_mean,
                "predicted_m2": pred_m2, "reference_m2": gt_m2, "cross_deviation_sum": covariance,
            },
        }

    @classmethod
    def evaluate_unreferenced_scene(cls, dsm, dtm, structural_heights, terrain_type="Uploaded Scene"):
        """Descriptive reconstruction values cannot establish accuracy without reference data."""
        result = cls._unassessed(terrain_type, "NO_INDEPENDENT_REFERENCE")
        result["mode"] = "UNREFERENCED_RECONSTRUCTION"
        try:
            arrays = [np.ma.asarray(a, dtype=float) for a in (dsm, dtm, structural_heights)]
            if any(a.shape != arrays[0].shape for a in arrays) or arrays[0].ndim != 2:
                return result
            values = [np.ma.getdata(a) for a in arrays]
            masks = [np.ma.getmaskarray(a) for a in arrays]
            mask = np.logical_and.reduce([
                np.isfinite(value) & ~hidden_mask & (value != -9999) & (value != -32768)
                for value, hidden_mask in zip(values, masks)
            ])
            valid = int(np.count_nonzero(mask))
            result.update(total_sample_count=int(arrays[0].size), valid_sample_count=valid,
                          invalid_sample_count=int(mask.size) - valid, sample_points_evaluated=0)
            if valid:
                heights = values[2][mask]
                structures = heights[heights > 1.5]
                result["reconstruction_statistics"] = {
                    "mean_structural_value": cls._number(np.mean(structures), 2) if structures.size else None,
                    "max_structural_value": cls._number(np.max(heights), 2),
                    "structural_dispersion": cls._number(np.std(heights), 2),
                    "surface_value_span": cls._number(np.ptp(values[0][mask]), 2),
                    "units": "surface units; metric scale must be established separately",
                }
        except (TypeError, ValueError):
            pass
        return result

    @classmethod
    def evaluate_scene(cls, predicted_dsm, scene, terrain_type=None):
        """Keep raw-label nodata support separate from any assumed-base DSM values."""
        support = None
        raw_agl = scene.get("ground_truth_agl")
        if raw_agl is not None:
            try:
                labels = np.ma.asarray(raw_agl, dtype=float)
                raw = np.ma.getdata(labels)
                support = np.isfinite(raw) & ~np.ma.getmaskarray(labels) & (raw != -9999) & (raw != -32768)
            except (TypeError, ValueError):
                return cls._unassessed(terrain_type or scene.get("terrain_type", "Scene"), "INVALID_RAW_REFERENCE_SUPPORT")
        metrics = cls.evaluate(predicted_dsm, scene.get("ground_truth_dsm"), terrain_type or scene.get("terrain_type", "Scene"),
                               scene.get("geo_metadata", {}).get("gsd_m"), valid_mask=support)
        metrics["reference_support_rule"] = "Finite, unmasked reference values; raw AGL nodata excluded before adding any base. Negative nonsentinel labels are retained, not clipped."
        metrics["raw_reference_invalid_sample_count"] = int(np.count_nonzero(~support)) if support is not None else None
        return metrics

    @staticmethod
    def scene_provenance(scene, scene_id=""):
        """Known procedural fixtures always remain synthetic, including legacy aliases."""
        identity = (scene_id + " " + str(scene.get("scene_id", ""))).lower()
        fixture = any(token in identity for token in ("sac", "ahmedabad", "hilly", "ridge", "himalaya"))
        declared = scene.get("is_synthetic")
        synthetic = True if fixture or declared is True else (False if declared is False or "gamus_dc_" in identity else None)
        group = "synthetic" if synthetic is True else ("real" if synthetic is False else "unknown")
        supplied = scene.get("provenance") or {}
        supplied = supplied if isinstance(supplied, dict) else {"source_description": str(supplied)}
        result = {
            **supplied, "data_kind": group, "is_synthetic": synthetic,
            "source_description": (
                "Procedurally generated RGB and reference surface; demonstration fixture, not SAC imagery or CartoDEM."
                if synthetic is True else
                "Bundled GAMUS RGB/AGL pair; reference DSM uses an assumed base elevation. Absolute datum is not independently validated."
                if synthetic is False and "gamus_dc_" in identity else
                supplied.get("source_description", "Source declared non-procedural; independent absolute-datum validation is not established.")
                if synthetic is False else "Source provenance is unverified; excluded from real-data performance aggregation."
            ),
            "reference_kind": "PROCEDURAL_REFERENCE" if synthetic is True else "BASE_PLUS_AGL_REFERENCE" if synthetic is False else "UNVERIFIED_REFERENCE",
            "absolute_datum_validated": False,
            "georeferencing_status": "UNREFERENCED" if scene.get("geo_metadata", {}).get("crs") == "UNREFERENCED" else "ILLUSTRATIVE_METADATA" if synthetic is True else "BUNDLED_METADATA_UNVERIFIED",
            "counts_as_real_performance": synthetic is False,
            "reference_provenance": scene.get("reference_provenance", {}),
            "reference_surface_semantics": scene.get("reference_surface_semantics", "UNVERIFIED_ABSOLUTE_DATUM"),
        }
        return DepthWizardBenchmark._json_safe(result)

    @classmethod
    def _aggregate(cls, evaluations):
        metrics = [item["metrics"] for item in evaluations]
        macro, metric_counts = {}, {}
        for key in cls.METRIC_KEYS:
            values = [m[key] for m in metrics if m.get(key) is not None and np.isfinite(m[key])]
            metric_counts[key] = len(values)
            macro[key] = cls._number(np.mean(values), 4 if key in ("pearson_correlation_r", "r_squared") else 2) if values else None
        samples = sum(int(m.get("valid_sample_count", 0)) for m in metrics)
        total = sum(int(m.get("total_sample_count", 0)) for m in metrics)
        sufficient = [m["sufficient_statistics"] for m in metrics if m.get("sufficient_statistics")]
        pooled = {key: None for key in cls.METRIC_KEYS}
        # Parallel variance combination retains between-scene variance; averaging r does not.
        count, pm, gm, p_m2, g_m2, cross = 0, 0.0, 0.0, 0.0, 0.0, 0.0
        for stat in sufficient:
            n = stat["count"]
            combined = count + n
            dp, dg = stat["predicted_mean"] - pm, stat["reference_mean"] - gm
            factor = count * n / combined
            p_m2 += stat["predicted_m2"] + dp * dp * factor
            g_m2 += stat["reference_m2"] + dg * dg * factor
            cross += stat["cross_deviation_sum"] + dp * dg * factor
            pm += dp * n / combined
            gm += dg * n / combined
            count = combined
        if count:
            sse = sum(s["squared_error_sum"] for s in sufficient)
            pooled.update(
                rmse_meters=cls._number(np.sqrt(sse / count), 2),
                mae_meters=cls._number(sum(s["absolute_error_sum"] for s in sufficient) / count, 2),
                bias_meters=cls._number(sum(s["error_sum"] for s in sufficient) / count, 2),
                r_squared=cls._number(1 - sse / g_m2, 4) if g_m2 > 0 else None,
                pearson_correlation_r=cls._number(np.clip((cross / np.sqrt(p_m2)) / np.sqrt(g_m2), -1, 1), 4) if p_m2 > 0 and g_m2 > 0 else None,
            )
        return {
            "evaluated_scene_count": len(evaluations), "valid_sample_count": samples,
            "total_sample_count": total, "invalid_sample_count": total - samples,
            "macro_metrics": macro, "macro_metric_scene_counts": metric_counts,
            "macro_definition": "Unweighted mean of defined per-scene metrics; each scene has equal weight.",
            "pooled_metrics": pooled,
            "pooled_definition": "Joint valid-pixel errors and moments; each valid pixel has equal weight.",
            "pooled_limitations": "Pooled percentiles, NMAD, dispersion and max error are not reconstructed from per-scene summaries and remain null.",
        }

    @classmethod
    def summarize(cls, evaluations, failed_scenes, attempted_count):
        groups = {
            kind: cls._aggregate([item for item in evaluations if item["provenance"]["data_kind"] == kind])
            for kind in ("real", "synthetic", "unknown")
        }
        partial = any(item["metrics"]["status"] == "PARTIAL" for item in evaluations)
        status = "FAILED" if not evaluations else "PARTIAL" if failed_scenes or partial or len(evaluations) != attempted_count else "SUCCESS"
        real = groups["real"]
        macro = real["macro_metrics"]
        summary = {
            "certification": "NOT CERTIFIED", "complete": status == "SUCCESS",
            "aggregation_scope": "Real-data scenes only. Procedural and unknown-provenance results are reported separately.",
            "attempted_scenes_count": attempted_count, "evaluated_scenes_count": len(evaluations),
            "failed_scenes_count": len(failed_scenes),
            "partially_evaluated_scenes_count": sum(item["metrics"]["status"] == "PARTIAL" for item in evaluations),
            "real_evaluated_scenes_count": groups["real"]["evaluated_scene_count"],
            "synthetic_evaluated_scenes_count": groups["synthetic"]["evaluated_scene_count"],
            "unknown_source_evaluated_scenes_count": groups["unknown"]["evaluated_scene_count"],
            "valid_sample_count": real["valid_sample_count"],
            "all_source_valid_sample_count": sum(g["valid_sample_count"] for g in groups.values()),
            "average_rmse_meters": macro["rmse_meters"], "average_mae_meters": macro["mae_meters"],
            "average_correlation_r": macro["pearson_correlation_r"],
            "average_le90_meters": macro["le90_meters"], "average_nmad_meters": macro["nmad_meters"],
            "macro_metrics": macro, "pooled_metrics": real["pooled_metrics"],
            "macro_metric_scene_counts": real["macro_metric_scene_counts"],
        }
        matrix = [{
            "scene_id": item["scene_id"], "category": item["landscape_category"],
            "label": item["landscape_label"], "status": item["metrics"]["status"],
            "rmse_m": item["metrics"]["rmse_meters"], "mae_m": item["metrics"]["mae_meters"],
            "pearson_r": item["metrics"]["pearson_correlation_r"],
            "le90_m": item["metrics"]["le90_meters"], "nmad_m": item["metrics"]["nmad_meters"],
            "valid_sample_count": item["metrics"]["valid_sample_count"], "provenance": item["provenance"],
        } for item in evaluations]
        matrix.extend({"scene_id": item["scene_id"], "category": item["landscape_category"],
                       "label": item["landscape_label"], "status": "FAILED", "reason": item["reason"],
                       "rmse_m": None, "mae_m": None, "pearson_r": None, "le90_m": None, "nmad_m": None,
                       "valid_sample_count": 0, "provenance": item["provenance"]} for item in failed_scenes)
        return {
            "status": status, "certification": "NOT CERTIFIED", "benchmark_summary": summary,
            "benchmark_results": {**macro, "status": status, "certification": "NOT CERTIFIED", "aggregation": "real-data macro"},
            "provenance_groups": groups, "landscape_stability_matrix": matrix,
            "performance_matrix": matrix, "scene_evaluations": evaluations, "failed_scenes": failed_scenes,
        }

    @classmethod
    def run_suite(cls, engine, terrain_specs=None):
        """Shared HTTP/CLI runner; reference arrays are used only after inference."""
        specs = cls.TERRAIN_SPECS if terrain_specs is None else terrain_specs
        evaluations, failures = [], []
        for spec in specs:
            provenance = cls.scene_provenance({}, spec["id"])
            calibration = {}
            try:
                scene = engine.load_gamus_scene(spec["id"], resample_size=512)
                provenance = cls.scene_provenance(scene, spec["id"])
                relative = engine.extract_relative_depth(scene["rgb_image"])
                meta = scene.get("geo_metadata", {})
                # Bundled fixture bounds are illustrative/unverified, not geodetic controls.
                calibrated = engine.calibrate_to_absolute_dsm(
                    rel_depth=relative, base_srtm_elevation_m=scene["base_elevation_m"],
                    max_structural_height_m=scene["max_structural_height_m"],
                    gsd_m=meta.get("gsd_m", 0.6), geo_bounds=None,
                )
                stats = calibrated.get("stats", {})
                calibration = cls._json_safe({key: stats.get(key) for key in (
                    "status", "surface_type", "surface_semantics", "is_metric", "dtm_source", "control_status",
                    "calibration_status", "height_scale_status", "terrain_status", "vertical_reference_status",
                    "horizontal_reference_status", "terrain_provenance", "inference_provenance", "refusal_reason",
                )})
                if stats.get("status") == "NOT_ASSESSED" or calibrated.get("status") == "NOT_ASSESSED":
                    raise ValueError(stats.get("reason") or calibrated.get("reason") or "TERRAIN_NOT_ASSESSED")
                if stats.get("is_metric") is not True:
                    raise ValueError(stats.get("refusal_reason") or "METRIC_SCALE_NOT_ESTABLISHED")
                metrics = cls.evaluate_scene(calibrated["dsm"], scene, spec["label"])
                if metrics["status"] == "NOT_ASSESSED":
                    raise ValueError(metrics["reason"])
                evaluations.append({
                    "scene_id": spec["id"], "landscape_category": spec["cat"], "landscape_label": spec["label"],
                    "scene_name": scene["name"], "metrics": metrics, "provenance": provenance,
                    "calibration": calibration,
                })
            except Exception as exc:
                failures.append({"scene_id": spec["id"], "landscape_category": spec["cat"],
                                 "landscape_label": spec["label"], "status": "FAILED",
                                 "reason": str(exc), "error_type": type(exc).__name__, "provenance": provenance,
                                 "calibration": calibration})
        return cls._json_safe(cls.summarize(evaluations, failures, len(specs)))
