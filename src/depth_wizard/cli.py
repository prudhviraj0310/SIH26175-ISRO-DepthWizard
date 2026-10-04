"""
DepthWizard Command-Line Interface (CLI)
========================================
Research prototype command-line utility for SIH26175.
Features:
- Single-view height estimation on arbitrary GeoTIFF / PNG / JPG
- 4-Landscape descriptive benchmark execution
- Public elevation surface-reference sampling (not certified DTM anchoring)
- 32-bit floating point GeoTIFF export
- 3D Wavefront OBJ mesh generation
"""

import sys
import os
import argparse
import json
import tempfile
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.depth_wizard.elevation_engine import ElevationEngine, HAS_TORCH_TRANSFORMERS, HAS_RASTERIO
from src.depth_wizard.benchmark import DepthWizardBenchmark
from src.depth_wizard.case_store import json_safe

def print_banner():
    banner = r"""
  ╔═══════════════════════════════════════════════════════════════════╗
  ║   SIH26175 — DEPTHWIZARD RESEARCH PROTOTYPE CLI                 ║
  ║   Monocular Height Estimation & Cross-Landscape Benchmark Suite  ║
  ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def get_system_info():
    print_banner()
    print("── SYSTEM ENVIRONMENT ──────────────────────────────────────────")
    print(f" Python Version      : {sys.version.split()[0]}")
    
    device = "CPU (Fallback)"
    if HAS_TORCH_TRANSFORMERS:
        import torch
        if torch.backends.mps.is_available():
            device = "Apple Silicon MPS (Hardware Accelerated)"
        elif torch.cuda.is_available():
            device = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    print(f" Compute Engine      : {device}")
    print(f" Foundation Model    : Depth Anything V2 Small (cached availability is disclosed per case)")
    print(f" Geospatial Rasterio : {'✓ Available (32-bit GeoTIFF + CRS)' if HAS_RASTERIO else '✗ Missing'}")
    print(f" SRTM Provider       : Copernicus GLO-30 (Open-Meteo) + SRTM-30 (Open-Elevation)")
    print("────────────────────────────────────────────────────────────────")

def run_benchmark():
    print_banner()
    print("── DESCRIPTIVE LANDSCAPE BENCHMARK ─────────────────────────────")
    report = DepthWizardBenchmark.run_suite(ElevationEngine())

    def value(number, digits=2):
        return f"{number:.{digits}f}" if number is not None else "unassessed"

    for scene in report["scene_evaluations"]:
        metrics = scene["metrics"]
        print(f"{scene['landscape_label']} [{scene['provenance']['data_kind']}]: {metrics['status']}; valid samples={metrics['valid_sample_count']}/{metrics['total_sample_count']}")
        print(f"  RMSE={value(metrics['rmse_meters'])} m; MAE={value(metrics['mae_meters'])} m; r={value(metrics['pearson_correlation_r'], 4)}; LE90={value(metrics['le90_meters'])} m; NMAD={value(metrics['nmad_meters'])} m")
        print(f"  Source: {scene['provenance']['source_description']}")
    for failure in report["failed_scenes"]:
        print(f"{failure['landscape_label']} [{failure['provenance']['data_kind']}]: FAILED — {failure['reason']}")
    for source, group in report["provenance_groups"].items():
        macro, pooled = group["macro_metrics"], group["pooled_metrics"]
        print(f"{source.upper()}: {group['evaluated_scene_count']} scenes, {group['valid_sample_count']} valid samples")
        print(f"  Macro (equal scene weight): RMSE={value(macro['rmse_meters'])} m; MAE={value(macro['mae_meters'])} m; r={value(macro['pearson_correlation_r'], 4)}")
        print(f"  Pooled (equal valid-pixel weight): RMSE={value(pooled['rmse_meters'])} m; MAE={value(pooled['mae_meters'])} m; r={value(pooled['pearson_correlation_r'], 4)}")
    summary = report["benchmark_summary"]
    print(f"STATE: {report['status']} — {summary['evaluated_scenes_count']}/{summary['attempted_scenes_count']} scenes evaluated; {summary['failed_scenes_count']} failed")
    print("NOT CERTIFIED — descriptive measurements only; synthetic fixtures are separate from real-data results.")
    return report


def process_image(input_path: str, output_path: str, base_elev: float = 50.0, max_h: float = 45.0, export_obj: bool = False, source_provenance=None):
    print_banner()
    print(f"── PROCESSING IMAGE: {input_path} ──")
    engine = ElevationEngine()

    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    with open(input_path, "rb") as f:
        file_bytes = f.read()

    processed = engine.process_image_file(
        file_bytes=file_bytes,
        filename=os.path.basename(input_path),
        is_georeferenced=input_path.lower().endswith((".tif", ".tiff")),
        base_srtm_elevation_m=base_elev,
        max_structural_height_m=max_h,
        target_resample_size=512
    )

    dsm = processed["dsm"]
    stats = processed["stats"]
    meta = processed.get("geo_metadata", {})

    if processed.get("status") == "NOT_ASSESSED" or stats.get("status") == "NOT_ASSESSED" or not np.all(np.isfinite(dsm)):
        print(f"NOT_ASSESSED: {processed.get('reason') or stats.get('reason') or 'Invalid terrain support'}")
        return processed
    provenance = source_provenance or {"data_kind": "uploaded", "is_synthetic": None,
                                      "source_description": "User-provided imagery; no independent accuracy reference.",
                                      "counts_as_real_performance": False}
    units = "m" if stats.get("is_metric") is True else "relative surface units"
    print(f"✓ Surface Reconstructed: {dsm.shape[0]}x{dsm.shape[1]} grid ({stats.get('surface_type', 'UNASSESSED')})")
    print(f"  Minimum Value    : {stats.get('min_elevation_m')} {units}")
    print(f"  Maximum Value    : {stats.get('max_elevation_m')} {units}")
    print(f"  Terrain Source   : {stats.get('dtm_source', 'UNASSESSED')}")
    print(f"  Control Status   : {stats.get('height_scale_status', stats.get('control_status', 'UNASSESSED'))}")
    print(f"  Source           : {provenance['source_description']}")
    print(f"  NOT CERTIFIED    : {stats.get('refusal_reason') or 'No operational certification has been performed.'}")
    manifest = json_safe({"provenance": provenance, "calibration": stats, "geo_metadata": meta, "certification": "NOT CERTIFIED"})

    # Export GeoTIFF
    if output_path:
        tif_bytes = engine.export_dsm_geotiff(dsm, geo_meta=meta)
        with open(output_path, "wb") as f_out:
            f_out.write(tif_bytes)
        with open(output_path + ".metadata.json", "w") as metadata_file:
            json.dump(manifest, metadata_file, allow_nan=False, indent=2)
        print(f"✓ TIFF Exported    : {output_path} (source/control metadata: {output_path}.metadata.json)")

    # Export OBJ
    if export_obj:
        obj_path = os.path.splitext(output_path)[0] + ".obj" if output_path else "mesh.obj"
        obj_bytes = engine.export_dsm_obj(dsm, step=4)
        obj_str = obj_bytes.decode("utf-8") if isinstance(obj_bytes, bytes) else str(obj_bytes)
        with open(obj_path, "w") as f_obj:
            f_obj.write("# DepthWizard metadata: " + json.dumps(manifest, allow_nan=False) + "\n")
            f_obj.write(obj_str)
        print(f"✓ 3D Mesh Exported : {obj_path} (Wavefront OBJ)")

def main():
    parser = argparse.ArgumentParser(
        description="DepthWizard research CLI — Single-View Height Estimation & 3D Flythrough (SIH26175)"
    )
    parser.add_argument("--info", action="store_true", help="Display system runtime and GPU accelerator info")
    parser.add_argument("--benchmark", action="store_true", help="Run the 4-Landscape descriptive benchmark")
    parser.add_argument("--input", type=str, help="Path to input satellite optical image (GeoTIFF, PNG, JPG)")
    parser.add_argument("--output", type=str, default="output_dsm.tif", help="Path for exported GeoTIFF DSM")
    parser.add_argument("--base-elevation", type=float, default=50.0, help="Assumed base elevation prior (not a surveyed control)")
    parser.add_argument("--max-height", type=float, default=45.0, help="Assumed structural-height prior (not measured ground truth)")
    parser.add_argument("--obj", action="store_true", help="Also export 3D Wavefront OBJ mesh")
    parser.add_argument("--scene", type=str, help="Select and process a bundled benchmark scene (e.g. isro_sac_ahmedabad)")

    args = parser.parse_args()

    if args.info:
        get_system_info()
    elif args.benchmark:
        report = run_benchmark()
        return {"SUCCESS": 0, "PARTIAL": 2, "FAILED": 1}.get(report["status"], 1)
    elif args.input:
        process_image(args.input, args.output, args.base_elevation, args.max_height, args.obj)
    elif args.scene:
        engine = ElevationEngine()
        scene = engine.load_gamus_scene(args.scene)
        import cv2
        with tempfile.TemporaryDirectory(prefix="depthwizard-") as temp_dir:
            temp_img_path = os.path.join(temp_dir, "scene.png")
            cv2.imwrite(temp_img_path, cv2.cvtColor(scene["rgb_image"], cv2.COLOR_RGB2BGR))
            process_image(temp_img_path, args.output, scene["base_elevation_m"], scene["max_structural_height_m"], args.obj,
                          source_provenance=DepthWizardBenchmark.scene_provenance(scene, args.scene))
    else:
        parser.print_help()

if __name__ == "__main__":
    raise SystemExit(main())
