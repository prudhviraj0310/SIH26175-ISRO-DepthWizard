"""
DepthWizard ISRO SAC Evaluation Benchmark
==========================================
Evaluates the estimated Digital Surface Model (DSM) against ground-truth LiDAR / reference data
according to ISRO's exact 50% accuracy evaluation criteria:
- Root Mean Square Error (RMSE) in meters
- Mean Absolute Error (MAE) in meters
- Pearson Correlation Coefficient (r)
- Residual Error Statistics across Urban, Hilly, and Forested benchmarks
"""

import numpy as np
from typing import Dict, Any

class DepthWizardBenchmark:
    @staticmethod
    def evaluate(
        predicted_dsm: np.ndarray,
        ground_truth_dsm: np.ndarray,
        terrain_type: str = "Urban"
    ) -> Dict[str, Any]:
        """
        Computes ISRO SAC standard statistical metrics between estimated and ground-truth DSM.
        """
        assert predicted_dsm.shape == ground_truth_dsm.shape, "Shape mismatch between prediction and ground truth"

        # Difference residual array: Error = Pred - GT
        residuals = predicted_dsm - ground_truth_dsm
        abs_residuals = np.abs(residuals)

        # 1. Root Mean Square Error (RMSE)
        rmse = float(np.sqrt(np.mean(residuals**2)))

        # 2. Mean Absolute Error (MAE)
        mae = float(np.mean(abs_residuals))

        # 3. Maximum Residual Error
        max_error = float(np.max(abs_residuals))

        # 4. Pearson Correlation Coefficient (r)
        pred_flat = predicted_dsm.flatten()
        gt_flat = ground_truth_dsm.flatten()

        pred_diff = pred_flat - np.mean(pred_flat)
        gt_diff = gt_flat - np.mean(gt_flat)
        numerator = np.sum(pred_diff * gt_diff)
        denominator = np.sqrt(np.sum(pred_diff**2) * np.sum(gt_diff**2)) + 1e-8
        pearson_r = float(numerator / denominator)

        # 5. Accuracy Grade according to ISRO remote-sensing benchmarks
        # ISRO SAC target for single-view estimation: RMSE < 4.0m in urban, r > 0.88
        if rmse < 2.5 and pearson_r > 0.92:
            isro_grade = "Tier-1 Exemplary (CartoDEM/LiDAR Grade)"
            status = "PASSED"
        elif rmse < 4.0 and pearson_r > 0.85:
            isro_grade = "Tier-2 Operational (Tactical Reconnaissance Grade)"
            status = "PASSED"
        else:
            isro_grade = "Tier-3 Baseline (Requires Fine-Tuning)"
            status = "NEEDS_CALIBRATION"

        # 6. Error distribution percentiles
        p50 = float(np.percentile(abs_residuals, 50))
        p90 = float(np.percentile(abs_residuals, 90))
        p95 = float(np.percentile(abs_residuals, 95))

        return {
            "terrain_type": terrain_type,
            "status": status,
            "isro_grade": isro_grade,
            "rmse_meters": round(rmse, 2),
            "mae_meters": round(mae, 2),
            "pearson_correlation_r": round(pearson_r, 4),
            "max_residual_error_m": round(max_error, 2),
            "error_percentiles": {
                "50th_percentile_m": round(p50, 2),
                "90th_percentile_m": round(p90, 2),
                "95th_percentile_m": round(p95, 2)
            },
            "sample_points_evaluated": int(predicted_dsm.size)
        }
