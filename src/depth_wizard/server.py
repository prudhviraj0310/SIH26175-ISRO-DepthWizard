"""DepthWizard prototype API with owned cases and descriptive assessments."""

import json
import re
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Literal

import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, Form, Response
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.depth_wizard.elevation_engine import ElevationEngine
from src.depth_wizard.mesh_generator import MeshGenerator
from src.depth_wizard.benchmark import DepthWizardBenchmark
from src.depth_wizard.case_store import CaseStore, json_safe

app = FastAPI(
    title="DepthWizard 3D Flythrough Research Prototype",
    description="Single-view surface reconstruction and descriptive evaluation (SIH26175)",
    version="2.0.0",
)
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/") or request.url.path in {"/", "/index.html"}:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
engine = ElevationEngine()
mesh_gen = MeshGenerator(target_grid_size=128)
ENGINE_LOCK = threading.RLock()
CASE_STORE = CaseStore()
SESSION_COOKIE = "depthwizard_session"


class SceneSelectRequest(BaseModel):
    scene_id: str = Field(min_length=1, max_length=120)
    surface_source: Literal["ai_dsm", "lidar_gt"] = "ai_dsm"


class MeasurementRequest(BaseModel):
    p1_x: int = Field(ge=0)
    p1_y: int = Field(ge=0)
    p2_x: int = Field(ge=0)
    p2_y: int = Field(ge=0)
    ground_res_m: float = Field(default=0.5, gt=0, allow_inf_nan=False)


class CoordinateQuery(BaseModel):
    pixel_x: int
    pixel_y: int


class FloodSimulationRequest(BaseModel):
    water_level_m: Optional[float] = Field(default=None, allow_inf_nan=False)
    offset_m: Optional[float] = Field(default=2.0, allow_inf_nan=False)


class HLZRequest(BaseModel):
    pad_radius_m: float = Field(default=8.0, gt=0, allow_inf_nan=False)
    max_slope_deg: float = Field(default=5.0, gt=0, le=90, allow_inf_nan=False)


def _error(code, message, status_code=422, **details):
    return JSONResponse(status_code=status_code, content=json_safe({
        "status": "NOT_ASSESSED", "error_code": code, "message": message,
        "certification": "NOT CERTIFIED", **details,
    }))


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # FastAPI's default error response can itself fail to serialize a NaN input.
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return _error("INVALID_REQUEST", "Request values are invalid.", errors=errors)


def _nonfinite(value):
    if isinstance(value, Mapping):
        return any(_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_nonfinite(item) for item in value)
    if isinstance(value, np.ndarray):
        return not np.all(np.isfinite(value))
    if isinstance(value, (float, np.floating)):
        return not np.isfinite(value)
    return False


def _valid_terrain(processed):
    if not isinstance(processed, Mapping) or not isinstance(processed.get("stats", {}), Mapping):
        raise ValueError("Terrain assessment metadata is invalid.")
    stats = processed.get("stats", {})
    if processed.get("status") == "NOT_ASSESSED" or stats.get("status") == "NOT_ASSESSED":
        raise ValueError(processed.get("reason") or stats.get("reason") or "Terrain was not assessed.")
    arrays = [np.ma.asarray(processed.get(key), dtype=float) for key in ("dsm", "dtm", "structural_heights")]
    if any(a.ndim != 2 or min(a.shape, default=0) < 2 for a in arrays):
        raise ValueError("Terrain requires nonempty two-dimensional grids.")
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Terrain grids have inconsistent shapes.")
    if any(np.any(np.ma.getmaskarray(a)) or not np.all(np.isfinite(a))
            or np.any(np.isin(np.ma.getdata(a), [-9999, -32768])) for a in arrays):
        raise ValueError("Terrain contains invalid or nodata support.")
    if np.any(arrays[2] < 0) or not np.allclose(arrays[0], arrays[1] + arrays[2], rtol=1e-6, atol=0.05):
        raise ValueError("Terrain decomposition is inconsistent: DSM must equal DTM plus structural heights.")


def _case(request):
    explicit = request.query_params.get("case_id")
    if explicit is None:
        explicit = request.headers.get("X-DepthWizard-Case")
    try:
        return CASE_STORE.get(request.cookies.get(SESSION_COOKIE), explicit)
    except KeyError as exc:
        code = exc.args[0]
        if explicit is not None:
            return _error("CASE_NOT_FOUND", "Requested case is missing, expired, or not owned by this session.", 404)
        return _error(code, "Select or upload a scene for this client. Its case may have expired or been evicted.", 409)


def _case_metadata(case):
    data = case.data
    return json_safe({
        "case_id": case.case_id, "scene_id": data["scene_id"],
        "scene_name": data["scene_name"], "provenance": data["provenance"],
        "surface_source": data["surface_source"], "calibration": data["stats"],
        "geo_metadata": data["geo_metadata"], "benchmark": data["benchmark"],
        "certification": "NOT CERTIFIED",
    })


def _publish(request, data, response_data):
    owner = CASE_STORE.session(request.cookies.get(SESSION_COOKIE))
    try:
        case = CASE_STORE.put(owner, data)
    except KeyError:
        return _error("SESSION_EXPIRED", "Processing session expired; select the scene again.", 409)
    response = JSONResponse(content=json_safe({
        **response_data, "case_id": case.case_id, "case_expires_in_seconds": CASE_STORE.ttl_seconds,
        "provenance": data["provenance"], "calibration": data["stats"], "certification": "NOT CERTIFIED",
    }))
    response.set_cookie(SESSION_COOKIE, owner, max_age=int(CASE_STORE.ttl_seconds),
                        httponly=True, samesite="lax", secure=request.url.scheme == "https", path="/")
    response.headers["Cache-Control"] = "no-store"
    return response


def _metric_required(case):
    if case.data["stats"].get("is_metric") is not True:
        return _error("METRIC_SCALE_NOT_ESTABLISHED", case.data["stats"].get("refusal_reason") or
                      "Metric scale is not established for this case.", case_id=case.case_id,
                      provenance=case.data["provenance"], calibration=case.data["stats"])
    return None


def _case_resolution(case):
    resolution = case.data["stats"].get("gsd_m", case.data["geo_metadata"].get("gsd_m"))
    try:
        resolution = float(resolution)
    except (TypeError, ValueError):
        raise ValueError("Ground sampling distance is not established for this case.")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Ground sampling distance must be finite and positive.")
    return resolution


def _assessment(result, case):
    invalid = result.get("status") not in {"SUCCESS", "ASSESSED"} or _nonfinite(result)
    body = {**result, "case_id": case.case_id, "provenance": case.data["provenance"],
            "calibration": case.data["stats"], "certification": "NOT CERTIFIED"}
    if invalid:
        body["status"] = "NOT_ASSESSED"
        body.setdefault("reason", "INVALID_OR_UNAVAILABLE_ASSESSMENT")
        if "candidate_zones" in body or "zones" in body:
            body.update(candidate_zones=[], zones=[], count=0, detected_zones_count=0)
    return JSONResponse(status_code=422 if invalid else 200, content=json_safe(body))


def _generate_mesh(surface, rgb_image, stats, *, ground_truth_dsm=None, dtm=None):
    """Build a render payload without relabelling mesh failures as terrain failures."""
    try:
        mesh = mesh_gen.generate_mesh_payload(
            surface, rgb_image, stats, ground_truth_dsm=ground_truth_dsm, dtm=dtm,
        )
    except (ValueError, TypeError) as exc:
        return _error("INVALID_MESH", str(exc))
    if not isinstance(mesh, Mapping) or _nonfinite(mesh):
        return _error("INVALID_MESH", "Rendered surface has invalid or nonfinite support.")
    return mesh


@app.get("/")
async def index_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/health")
async def health_check():
    backbone = getattr(engine, "dav2", None)
    return {"status": "AVAILABLE", "health_scope": "SERVICE_AVAILABILITY_ONLY", "service_available": True,
            "operational_readiness": "NOT_ASSESSED", "metric_accuracy_status": "NOT_ASSESSED",
            "system": "DepthWizard Research Prototype", "ps_number": "SIH26175",
            "backbone": getattr(backbone, "model_id", "depth-anything/Depth-Anything-V2-Small-hf"),
            "backbone_status": getattr(backbone, "load_status", "UNAVAILABLE"),
            "version": "2.0.0", "certification": "NOT CERTIFIED"}


SCENE_CATALOG = (
    {"id": "isro_sac_ahmedabad", "name": "Synthetic Urban Campus (procedural fixture)", "terrain_type": "Synthetic Procedural Benchmark", "is_synthetic": True,
     "description": "Procedural buildings and RGB; not a scan of the SAC campus.", "base_elevation_m": 52.0, "max_structural_height_m": 25.0},
    {"id": "gamus_hilly_ridge", "name": "Synthetic High-Relief Ridge (procedural fixture)", "terrain_type": "Synthetic Procedural Benchmark", "is_synthetic": True,
     "description": "Procedural Gaussian ridge and RGB; no CartoDEM or stereo ground truth.", "base_elevation_m": 1150.0, "max_structural_height_m": 50.0},
    {"id": "gamus_dc_04_23", "name": "GAMUS DC_04_23: Commercial Core", "terrain_type": "Bundled RGB / AGL reference pair", "is_synthetic": False,
     "description": "Bundled GAMUS RGB/AGL; reference DSM adds an assumed base, not a validated absolute datum.", "base_elevation_m": 15.0, "max_structural_height_m": 25.0},
    {"id": "gamus_dc_11_33", "name": "GAMUS DC_11_33: Suburban & Transit", "terrain_type": "Bundled RGB / AGL reference pair", "is_synthetic": False,
     "description": "Bundled GAMUS RGB/AGL; reference DSM adds an assumed base, not a validated absolute datum.", "base_elevation_m": 15.0, "max_structural_height_m": 25.0},
    {"id": "gamus_dc_02_26", "name": "GAMUS DC_02_26: Residential & Canopy", "terrain_type": "Bundled RGB / AGL reference pair", "is_synthetic": False,
     "description": "Bundled GAMUS RGB/AGL; reference DSM adds an assumed base, not a validated absolute datum.", "base_elevation_m": 15.0, "max_structural_height_m": 25.0},
)


@app.get("/api/scenes")
async def get_available_scenes():
    return {"scenes": [{**s, "base_elevation_status": "ASSUMED_LOCAL_BASE", "height_scale_status": "ASSUMED_PRIOR",
                        "certification": "NOT CERTIFIED", "provenance": DepthWizardBenchmark.scene_provenance(s, s["id"])}
                       for s in SCENE_CATALOG]}


@app.post("/api/scene/select")
async def select_and_process_scene(req: SceneSelectRequest, request: Request):
    # Legacy IDs remain accepted by the engine, but loading failures are explicit.
    try:
        scene = engine.load_gamus_scene(req.scene_id, resample_size=512)
    except FileNotFoundError:
        return _error("SCENE_UNAVAILABLE", "Requested scene is not available.", 404)
    except Exception as exc:
        return _error("SCENE_UNAVAILABLE", str(exc), 503)
    try:
        provenance = DepthWizardBenchmark.scene_provenance(scene, req.scene_id)
        meta = {**scene.get("geo_metadata", {}), "georeferencing_status": provenance["georeferencing_status"]}
        # Inference provenance and its calibration belong to the same operation.
        with ENGINE_LOCK:
            relative = engine.extract_relative_depth(scene["rgb_image"])
            calibrated = engine.calibrate_to_absolute_dsm(
                rel_depth=relative, base_srtm_elevation_m=scene["base_elevation_m"],
                max_structural_height_m=scene["max_structural_height_m"],
                gsd_m=meta.get("gsd_m", 0.6), geo_bounds=None,
            )
        _valid_terrain(calibrated)
        dsm, dtm = calibrated["dsm"], calibrated["dtm"]
        stats = dict(calibrated.get("stats", {}))
        reference = scene.get("ground_truth_dsm")
        reference_available = (reference is not None and np.asarray(reference).shape == np.asarray(dsm).shape
                               and not np.any(np.ma.getmaskarray(reference))
                               and not _nonfinite(np.asarray(reference)) and not np.any(np.isin(reference, [-9999, -32768])))
        if scene.get("ground_truth_agl") is not None:
            raw = np.ma.asarray(scene["ground_truth_agl"], dtype=float)
            reference_available = (reference_available and raw.shape == np.asarray(dsm).shape
                                   and not np.any(np.ma.getmaskarray(raw)) and np.all(np.isfinite(raw))
                                   and not np.any(np.isin(raw, [-9999, -32768])))
        if stats.get("is_metric") is not True:
            bench = DepthWizardBenchmark.evaluate_unreferenced_scene(dsm, dtm, calibrated["structural_heights"], scene["terrain_type"])
            bench["reason"] = stats.get("refusal_reason") or "METRIC_SCALE_NOT_ESTABLISHED"
        else:
            bench = DepthWizardBenchmark.evaluate_scene(dsm, scene)
        bench["provenance"] = provenance
        active_surface = dsm
        structural = calibrated["structural_heights"]
        active_mode = "Monocular reconstruction (not certified)"
        if req.surface_source == "lidar_gt":
            if reference is None:
                return _error("REFERENCE_UNAVAILABLE", "No reference surface is available for this scene.", 422)
            active_surface = np.asarray(reference)
            if not reference_available:
                return _error("REFERENCE_UNAVAILABLE", "Reference surface has invalid support.", 422)
            structural = active_surface - dtm
            stats.update(surface_type="REFERENCE_SURFACE", is_metric=False,
                         height_scale_status="REFERENCE_ONLY_UNVERIFIED_DATUM",
                         refusal_reason="Reference DSM has procedural or assumed-base heights; absolute geodetic scale is not independently validated.")
            active_mode = "Procedural reference surface" if provenance["is_synthetic"] else "Bundled AGL + assumed-base reference"
        stats.update(model_mode=active_mode, provenance=provenance)
        meta.update(surface_type=stats.get("surface_type"), is_metric=stats.get("is_metric"),
                    vertical_datum=stats.get("vertical_datum", meta.get("vertical_datum", "UNKNOWN")),
                    height_calibration_status=stats.get("height_scale_status", stats.get("calibration_status", "UNASSESSED")))
        comparison_reference = reference if req.surface_source == "ai_dsm" and reference_available else None
        mesh = _generate_mesh(active_surface, scene["rgb_image"], stats,
                              ground_truth_dsm=comparison_reference, dtm=dtm)
        if isinstance(mesh, Response):
            return mesh
        mesh["provenance"] = provenance
        mesh["surface_source"] = req.surface_source
        mesh["reference_comparison_available"] = comparison_reference is not None
        # Immutable active geometry is also the geometry used by exports/measurements.
        data = {"scene_id": scene.get("scene_id", req.scene_id), "scene_name": scene["name"],
                "dsm": active_surface, "predicted_dsm": dsm, "dtm": dtm, "structural_heights": structural,
                "mesh_payload": mesh, "benchmark": bench, "geo_metadata": meta, "stats": stats,
                "provenance": provenance, "surface_source": req.surface_source}
        return _publish(request, data, {"status": "SUCCESS", "scene_name": scene["name"],
                        "terrain_type": scene["terrain_type"], "surface_source": req.surface_source,
                         "reference_comparison_available": comparison_reference is not None,
                        "active_mode_name": active_mode, "mesh_payload": mesh, "benchmark": bench,
                        "geo_metadata": meta, "territory": engine.verify_indian_territory(meta)})
    except (ValueError, TypeError, KeyError) as exc:
        return _error("INVALID_TERRAIN", str(exc), 422)
    except Exception as exc:
        return _error("PROCESSING_UNAVAILABLE", str(exc), 503)


def _export_headers(case, filename):
    return {
        "Content-Disposition": f'attachment; filename="{filename}"', "Cache-Control": "no-store",
        "X-DepthWizard-Case": case.case_id, "X-DepthWizard-Provenance": case.data["provenance"]["data_kind"],
        "X-DepthWizard-Surface-Source": case.data["surface_source"],
        "X-DepthWizard-Metric-Status": "METRIC" if case.data["stats"].get("is_metric") is True else "UNASSESSED",
        "X-DepthWizard-Certification": "NOT CERTIFIED",
        "Link": f'</api/export/metadata?case_id={case.case_id}>; rel="describedby"',
    }


def _annotate_tiff(content, metadata):
    # The installed optional rasterio dependency embeds provenance in the artifact.
    # Headers and the JSON export manifest remain available on minimal runtimes.
    try:
        import rasterio
        from rasterio.io import MemoryFile
    except ImportError:
        return content
    with MemoryFile() as memory:
        memory.write(content)
        with rasterio.open(memory.name, "r+") as dataset:
            dataset.update_tags(DEPTHWIZARD_METADATA=json.dumps(metadata, allow_nan=False, separators=(",", ":")))
        memory.seek(0)
        return memory.read()


@app.get("/api/export/metadata")
async def export_case_metadata(request: Request):
    case = _case(request)
    if isinstance(case, Response):
        return case
    return JSONResponse(content=_case_metadata(case), headers={"Cache-Control": "no-store"})


@app.get("/api/export/dsm")
async def export_active_dsm(request: Request):
    case = _case(request)
    if isinstance(case, Response):
        return case
    try:
        content = engine.export_dsm_geotiff(case.data["dsm"], geo_meta=case.data["geo_metadata"])
        content = _annotate_tiff(content, _case_metadata(case))
        scene = re.sub(r"[^a-zA-Z0-9_-]", "_", case.data["scene_id"])[:100]
        kind = case.data["provenance"]["data_kind"]
        return Response(content=content, media_type="image/tiff",
                        headers=_export_headers(case, f"DepthWizard_{kind}_{scene}.tif"))
    except (ValueError, TypeError) as exc:
        return _error("EXPORT_NOT_ASSESSED", str(exc))
    except Exception as exc:
        return _error("EXPORT_UNAVAILABLE", str(exc), 503)


@app.get("/api/export/obj")
async def export_active_obj(request: Request):
    case = _case(request)
    if isinstance(case, Response):
        return case
    try:
        content = engine.export_dsm_obj(case.data["dsm"], step=4)
        if isinstance(content, str):
            content = content.encode("utf-8")
        manifest = json.dumps(_case_metadata(case), allow_nan=False, separators=(",", ":"))
        content = ("# DepthWizard metadata: " + manifest + "\n").encode("utf-8") + content
        return Response(content=content, media_type="model/obj",
                        headers=_export_headers(case, f"depthwizard_{case.data['provenance']['data_kind']}_terrain.obj"))
    except (ValueError, TypeError) as exc:
        return _error("EXPORT_NOT_ASSESSED", str(exc))
    except Exception as exc:
        return _error("EXPORT_UNAVAILABLE", str(exc), 503)


@app.post("/api/disaster/flood")
async def simulate_flood_inundation(req: FloodSimulationRequest, request: Request):
    case = _case(request)
    if isinstance(case, Response):
        return case
    unavailable = _metric_required(case)
    if unavailable is not None:
        return unavailable
    water = req.water_level_m
    if water is None:
        water = float(np.min(case.data["dsm"])) + (req.offset_m if req.offset_m is not None else 2.0)
    try:
        result = engine.simulate_flood(case.data["dsm"], water, structural_heights=case.data["structural_heights"],
                                       ground_res_m=_case_resolution(case))
        return _assessment(result, case)
    except (ValueError, TypeError) as exc:
        return _error("FLOOD_NOT_ASSESSED", str(exc))
    except Exception as exc:
        return _error("FLOOD_UNAVAILABLE", str(exc), 503)


@app.post("/api/disaster/landing-zones")
@app.post("/api/disaster/landing_zones")
async def detect_helicopter_landing_zones(request: Request, req: Optional[HLZRequest] = None):
    case = _case(request)
    if isinstance(case, Response):
        return case
    unavailable = _metric_required(case)
    if unavailable is not None:
        return unavailable
    try:
        result = engine.detect_landing_zones(
            dsm=case.data["dsm"], dtm=case.data["dtm"], structural_heights=case.data["structural_heights"],
            pad_radius_m=req.pad_radius_m if req else 8.0, max_slope_deg=req.max_slope_deg if req else 5.0,
            ground_res_m=_case_resolution(case),
        )
        result = {**result, "count": result.get("detected_zones_count", len(result.get("candidate_zones", []))),
                  "zones": result.get("candidate_zones", [])}
        return _assessment(result, case)
    except (ValueError, TypeError) as exc:
        return _error("LANDING_ZONES_NOT_ASSESSED", str(exc))
    except Exception as exc:
        return _error("LANDING_ZONES_UNAVAILABLE", str(exc), 503)


@app.post("/api/disaster/landslide")
@app.post("/api/disaster/landslide_risk")
async def screen_landslide_hazard(request: Request):
    case = _case(request)
    if isinstance(case, Response):
        return case
    unavailable = _metric_required(case)
    if unavailable is not None:
        return unavailable
    try:
        return _assessment(engine.screen_landslide_risk(case.data["dtm"], ground_res_m=_case_resolution(case)), case)
    except (ValueError, TypeError) as exc:
        return _error("SLOPE_SCREENING_NOT_ASSESSED", str(exc))
    except Exception as exc:
        return _error("SLOPE_SCREENING_UNAVAILABLE", str(exc), 503)


@app.post("/api/measure")
async def measure_3d_laser(req: MeasurementRequest, request: Request):
    case = _case(request)
    if isinstance(case, Response):
        return case
    unavailable = _metric_required(case)
    if unavailable is not None:
        return unavailable
    dsm = case.data["dsm"]
    if max(req.p1_x, req.p2_x) >= dsm.shape[1] or max(req.p1_y, req.p2_y) >= dsm.shape[0]:
        return _error("POINT_OUTSIDE_CASE", "Measurement points must lie within this case's grid.")
    try:
        result = engine.measure_distance_between_points(
            dsm=dsm, p1=(req.p1_x, req.p1_y), p2=(req.p2_x, req.p2_y), ground_resolution_m=req.ground_res_m,
        )
        if result.get("status") == "NOT_ASSESSED" or _nonfinite(result):
            return _error("INVALID_MEASUREMENT", "Measurement is not available.", measurement=result, case_id=case.case_id)
        return JSONResponse(content=json_safe({"status": "SUCCESS", "measurement": result, "case_id": case.case_id,
                                              "provenance": case.data["provenance"], "certification": "NOT CERTIFIED"}))
    except (ValueError, TypeError) as exc:
        return _error("INVALID_MEASUREMENT", str(exc))
    except Exception as exc:
        return _error("MEASUREMENT_UNAVAILABLE", str(exc), 503)


async def run_full_benchmark():
    """Callable without HTTP context; CLI/isolated tests receive an ordinary dict."""
    return DepthWizardBenchmark.run_suite(engine)


@app.get("/api/benchmark")
@app.get("/api/benchmark/run")
async def benchmark_http():
    with ENGINE_LOCK:
        result = await run_full_benchmark()
    code = {"SUCCESS": 200, "PARTIAL": 207, "FAILED": 503}.get(result["status"], 503)
    return JSONResponse(status_code=code, content=json_safe(result), headers={"Cache-Control": "no-store"})


@app.post("/api/upload")
async def upload_satellite_image(
    request: Request, file: UploadFile = File(...), is_georeferenced: bool = Form(False),
    base_srtm_elevation_m: float = Form(15.0), max_structural_height_m: float = Form(45.0),
):
    if not np.isfinite(base_srtm_elevation_m) or not np.isfinite(max_structural_height_m) or max_structural_height_m <= 0:
        return _error("INVALID_REQUEST", "Elevation/height priors must be finite; height must be positive.")
    try:
        content = await file.read()
        with ENGINE_LOCK:
            processed = engine.process_image_file(
                file_bytes=content, filename=file.filename or "uploaded_scene.tif",
                is_georeferenced=is_georeferenced, base_srtm_elevation_m=base_srtm_elevation_m,
                max_structural_height_m=max_structural_height_m, target_resample_size=512,
            )
        _valid_terrain(processed)
        dsm, dtm, stats = processed["dsm"], processed["dtm"], dict(processed["stats"])
        provenance = {"data_kind": "uploaded", "is_synthetic": None, "counts_as_real_performance": False,
                      "source_description": "User-uploaded imagery; source authenticity and accuracy are not independently verified.",
                      "reference_kind": "NO_INDEPENDENT_REFERENCE", "absolute_datum_validated": False,
                      "georeferencing_status": "FILE_METADATA" if processed.get("is_georeferenced") else "UNREFERENCED"}
        stats["provenance"] = provenance
        meta = processed.get("geo_metadata", {})
        mesh = _generate_mesh(dsm, processed["rgb_image"], stats, dtm=dtm)
        if isinstance(mesh, Response):
            return mesh
        mesh.update(provenance=provenance, surface_source="ai_dsm", reference_comparison_available=False)
        bench = DepthWizardBenchmark.evaluate_unreferenced_scene(dsm, dtm, processed["structural_heights"], processed["terrain_type"])
        bench["provenance"] = provenance
        data = {"scene_id": processed["scene_id"], "scene_name": processed["name"], "dsm": dsm, "dtm": dtm,
                "structural_heights": processed["structural_heights"], "mesh_payload": mesh,
                "benchmark": bench, "geo_metadata": meta, "stats": stats, "provenance": provenance, "surface_source": "ai_dsm"}
        return _publish(request, data, {"status": "SUCCESS", "scene_name": processed["name"],
                        "terrain_type": processed["terrain_type"], "model_mode": processed["model_mode"],
                        "is_georeferenced": processed["is_georeferenced"], "territory": engine.verify_indian_territory(meta),
                        "mesh_payload": mesh, "benchmark": bench, "geo_metadata": meta})
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return _error("INVALID_TERRAIN", str(exc))
    except Exception as exc:
        return _error("PROCESSING_UNAVAILABLE", str(exc), 503)
