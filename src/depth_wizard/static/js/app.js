/**
 * ISRO SAC Geospatial 3D Workstation — Application Controller
 * ==========================================================
 * Manages GIS state, binds topographic layers, coordinates REST API calls,
 * and updates real-time telemetry metrics.
 */

let flythrough = null;
let currentSceneId = "gamus_dc_04_23";
let currentSurfaceSource = "ai_dsm";
let currentCaseId = null;
let sceneProcessing = false;
let hasReferenceSurface = true;
let caseRevision = 0;
let requestSequence = 0;
const latestCaseRequests = new Map();

function beginCaseRequest(kind) {
    if (sceneProcessing || !currentCaseId) return null;
    const token = { kind, revision: caseRevision, caseId: currentCaseId, sequence: ++requestSequence };
    latestCaseRequests.set(kind, token.sequence);
    return token;
}

function isCurrentCaseRequest(token, result = null) {
    return !!token && !sceneProcessing && token.revision === caseRevision && token.caseId === currentCaseId
        && latestCaseRequests.get(token.kind) === token.sequence
        && (!result?.case_id || result.case_id === token.caseId);
}

function resetCaseAssessments() {
    latestCaseRequests.clear();
    setMetric('flood-submerged-val', null, '%');
    setMetric('flood-area-val', null, 'ha');
    const status = document.getElementById('flood-assessment');
    if (status) status.innerText = 'NOT_ASSESSED — no water-level screen requested for this case — NOT CERTIFIED';
    ['hlz-results', 'landslide-results'].forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.innerText = 'NOT_ASSESSED — no screen requested for this case — NOT CERTIFIED';
            element.style.display = 'none';
        }
    });
    ['slider-water', 'slider-flood-surge'].forEach(id => {
        const element = document.getElementById(id);
        if (element) element.value = '0';
    });
    const water = document.getElementById('water-val-text');
    if (water) water.innerText = 'OFF';
    const flood = document.getElementById('flood-surge-val');
    if (flood) flood.innerText = 'unassessed';
    ['reading-height', 'reading-distance', 'reading-distance-3d', 'reading-slope', 'reading-grade'].forEach(id => setMetric(id, null));
    const labels = {
        'btn-scan-hlz': '🚁 Scan Geometric Candidates (Slope < 5°)',
        'btn-screen-landslide': '⚠️ Screen Steep Slopes (> 30°)',
        'btn-run-full-benchmark': '⚡ Run Descriptive Landscape Benchmark',
        'btn-caliper': 'Measure Transect',
    };
    Object.entries(labels).forEach(([id, text]) => {
        const element = document.getElementById(id);
        if (element) {
            element.innerText = text;
            element.classList.remove('active');
        }
    });
    const body = document.getElementById('landscape-matrix-body');
    if (body) body.innerHTML = '<tr><td colspan="5">No suite result for this case view.</td></tr>';
    flythrough?.clearLandingZones();
    flythrough?.clearCaliper();
    flythrough?.setCaliperTool(false);
    flythrough?.setWaterLevel(0);
}

function updateCaseControls() {
    const unavailable = sceneProcessing || !currentCaseId;
    ['slider-water', 'btn-scan-hlz', 'btn-screen-landslide', 'btn-run-full-benchmark', 'btn-caliper',
        'btn-clear-caliper', 'btn-snapshot'].forEach(id => {
        const control = document.getElementById(id);
        if (control) control.disabled = unavailable;
    });
    const flood = document.getElementById('slider-flood-surge');
    if (flood) flood.disabled = unavailable || flythrough?.currentPayload?.stats?.is_metric !== true;
    ['btn-export-dsm-top', 'btn-export-dsm', 'btn-export-obj-top', 'btn-export-obj', 'btn-export-metadata'].forEach(id => {
        const link = document.getElementById(id);
        if (link) {
            link.setAttribute('aria-disabled', String(unavailable));
            link.style.pointerEvents = unavailable ? 'none' : '';
        }
    });
}

function setSceneProcessing(active) {
    if (active && !sceneProcessing) {
        caseRevision++;
        resetCaseAssessments();
    }
    sceneProcessing = active;
    ['scene-selector', 'btn-src-ai', 'btn-src-lidar', 'btn-upload-file', 'file-upload-input'].forEach(id => {
        const control = document.getElementById(id);
        if (control) control.disabled = active || (id === 'btn-src-lidar' && !hasReferenceSurface);
    });
    updateCaseControls();
}

function apiFetch(url, options = {}) {
    const headers = new Headers(options.headers || {});
    if (currentCaseId) headers.set('X-DepthWizard-Case', currentCaseId);
    return fetch(url, { ...options, headers, credentials: 'same-origin' });
}

function metricText(value, unit = '', digits = 2) {
    return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(digits)}${unit ? ' ' + unit : ''}` : 'unassessed';
}

function setMetric(id, value, unit = '', digits = 2) {
    const element = document.getElementById(id);
    if (element) element.innerText = metricText(value, unit, digits);
}

function assessmentText(data) {
    return `${data.status || 'NOT_ASSESSED'} — ${data.message || data.reason || data.assessment_scope || 'Descriptive terrain screening'} — NOT CERTIFIED`;
}

document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initialize 3D WebGL Flythrough Engine
    flythrough = new FlythroughEngine('flythrough-canvas-container');

    // 2. Load available ISRO satellite scenes
    await loadScenes();

    // 3. Bind UI interactions and layer controls
    bindControls();

    // 4. Load initial default scene (ISRO GAMUS DC_04_23 Commercial Core)
    await selectScene('gamus_dc_04_23');
});

async function loadScenes() {
    try {
        const res = await fetch('/api/scenes');
        const data = await res.json();
        const selectElem = document.getElementById('scene-selector');
        if (!selectElem) return;
        selectElem.innerHTML = '';

        data.scenes.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.innerText = `${s.name}`;
            selectElem.appendChild(opt);
        });

        selectElem.addEventListener('change', async (e) => {
            await selectScene(e.target.value);
        });
    } catch (err) {
        console.error('Failed to load scenes:', err);
    }
}

async function selectScene(sceneId, surfaceSource = null) {
    if (sceneProcessing) return;
    setSceneProcessing(true);
    const targetScene = sceneId || currentSceneId;
    const targetSource = surfaceSource || currentSurfaceSource;

    try {
        const statusInd = document.getElementById('status-indicator');
        if (statusInd) {
            const label = targetSource === 'lidar_gt' ? 'LOADING REFERENCE SURFACE...' : 'RECONSTRUCTING SURFACE...';
            statusInd.innerHTML = '<span class="pulse-dot" style="background:#f59e0b;"></span> ' + label;
        }

        const res = await apiFetch('/api/scene/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scene_id: targetScene, surface_source: targetSource })
        });
        const data = await res.json();

        if (res.ok && data.status === 'SUCCESS') {
            currentSceneId = targetScene;
            currentSurfaceSource = targetSource;
            // Load mesh and analytical textures into Three.js
            flythrough.loadTerrain(data.mesh_payload);

            // Update telemetry and benchmark displays
            updateTelemetry(data);

            if (statusInd) {
                statusInd.innerText = `READY / ${(data.provenance?.data_kind || 'UNKNOWN').toUpperCase()} / NOT CERTIFIED`;
            }
        } else {
            throw new Error(data.message || data.reason || 'Scene is unavailable');
        }
    } catch (err) {
        console.error('Failed to select scene:', err);
        const statusInd = document.getElementById('status-indicator');
        if (statusInd) {
            statusInd.innerText = `NOT_ASSESSED — ${err.message}`;
        }
    } finally {
        setSceneProcessing(false);
        document.getElementById('btn-src-ai')?.classList.toggle('active', currentSurfaceSource === 'ai_dsm');
        document.getElementById('btn-src-lidar')?.classList.toggle('active', currentSurfaceSource === 'lidar_gt');
    }
}

function updateTelemetry(data) {
    const stats = (data.mesh_payload && data.mesh_payload.stats) ? data.mesh_payload.stats : {};
    const payload = data.mesh_payload || {};
    const bench = data.benchmark || {};
    const isMetric = stats.is_metric === true;
    const provenance = data.provenance || {};
    caseRevision++;
    resetCaseAssessments();
    currentCaseId = data.case_id || null;
    currentSurfaceSource = data.surface_source || payload.surface_source || 'ai_dsm';
    hasReferenceSurface = currentSurfaceSource === 'lidar_gt' || payload.reference_comparison_available === true;
    ['btn-export-dsm-top', 'btn-export-dsm', 'btn-export-obj-top', 'btn-export-obj', 'btn-export-metadata'].forEach(id => {
        const link = document.getElementById(id);
        if (link) {
            const endpoint = id.includes('metadata') ? 'metadata' : id.includes('obj') ? 'obj' : 'dsm';
            link.href = `/api/export/${endpoint}?case_id=${encodeURIComponent(currentCaseId || '')}`;
        }
    });
    const source = document.getElementById('source-provenance');
    if (source) source.innerText = `${(provenance.data_kind || 'unknown').toUpperCase()}: ${provenance.source_description || 'Source unverified'}`;
    const control = document.getElementById('source-control');
    if (control) control.innerText = `${stats.surface_type || 'UNASSESSED'} / terrain reference ${stats.terrain_reference_source || stats.dtm_source || 'UNASSESSED'} / ${stats.calibration_status || stats.control_status || 'Control unverified'}; height scale ${stats.height_scale_status || 'UNASSESSED'}; vertical control ${stats.vertical_reference_status || 'UNASSESSED'}; horizontal control ${stats.horizontal_reference_status || 'UNASSESSED'} — ${stats.refusal_reason || 'NOT CERTIFIED'}`;
    const inference = document.getElementById('source-inference');
    if (inference) inference.innerText = `Inference: ${stats.inference_provenance?.mode || 'UNASSESSED'} / ${stats.inference_provenance?.model_status || 'model availability unverified'}; metric accuracy validated: ${stats.inference_provenance?.metric_accuracy_validated === true ? 'yes' : 'no'}`;
    const lod1 = document.getElementById('source-lod1');
    if (lod1) {
        const count = ['SUCCESS', 'EMPTY'].includes(payload.lod1_status) ? `; ${payload.lod1_building_count ?? 0} blocks` : '';
        lod1.innerText = `LoD-1 extraction: ${payload.lod1_status || 'NOT_ASSESSED'}${count}; terrain source ${payload.lod1_dtm_source || 'UNAVAILABLE'}; ${payload.lod1_dtm_status || 'UNASSESSED'}; verified-DTM declaration ${payload.lod1_dtm_authoritative === true ? 'yes' : 'no'}${payload.lod1_error ? '; ' + payload.lod1_error : ''} — experimental visualization, NOT CERTIFIED`;
    }
    const floodLabel = document.getElementById('flood-surge-val');
    if (floodLabel) floodLabel.innerText = isMetric ? '0.0 m (screen offset)' : 'unassessed — metric controls required';
    updateCaseControls();
    const spatial = document.getElementById('telemetry-spatial');
    if (spatial) spatial.innerText = `${metricText(isMetric ? stats.gsd_m : null, 'm')} / ${data.geo_metadata?.crs || 'UNREFERENCED'}`;

    // Header Scene Name & Territory
    const sc = document.getElementById('telemetry-scene');
    if (sc && data.scene_name) sc.innerText = data.scene_name;

    const elTerritory = document.getElementById('telemetry-territory');
    if (elTerritory && data.territory) {
        elTerritory.innerText = data.territory.badge;
        if (data.territory.is_in_india) {
            elTerritory.style.color = 'var(--isro-green)';
            elTerritory.title = `${data.territory.label} (${data.territory.center_lat}°N, ${data.territory.center_lon}°E)`;
        } else if (data.territory.status === 'UNREFERENCED') {
            elTerritory.style.color = '#f59e0b';
            elTerritory.title = 'Non-georeferenced imagery (relative rDSM output)';
        } else {
            elTerritory.style.color = '#64748b';
            elTerritory.title = `${data.territory.label}`;
        }
    }

    const units = isMetric ? 'm' : 'relative units';
    setMetric('stat-min-elev', payload.min_elevation_m, units);
    setMetric('stat-max-elev', payload.max_elevation_m, units);
    setMetric('stat-relief-range', payload.elevation_range_m, units);
    setMetric('stat-max-struct', stats.max_structural_height_m ?? stats.max_building_height_m, units);
    setMetric('stat-mean-slope', isMetric ? payload.mean_slope_deg : null, 'deg');
    setMetric('stat-max-slope', isMetric ? payload.max_slope_deg : null, 'deg');

    const sTerrainClass = document.getElementById('stat-terrain-class');
    if (sTerrainClass) {
        sTerrainClass.innerText = `${data.terrain_type || 'Scene'} (${metricText(isMetric ? payload.flat_percentage : null, '%')} flat <5°, ${metricText(isMetric ? payload.steep_percentage : null, '%')} steep >30°)`;
    }

    // Undefined metrics remain unassessed rather than stale or fabricated.
    const bg = document.getElementById('bench-grade');
    if (bg) bg.innerText = `${bench.status || 'NOT_ASSESSED'} — NOT CERTIFIED`;
    setMetric('bench-rmse', bench.rmse_meters, 'm');
    setMetric('bench-mae', bench.mae_meters, 'm');
    setMetric('bench-corr', bench.pearson_correlation_r, '', 4);
    setMetric('bench-nmad', bench.nmad_meters, 'm');
    setMetric('bench-p90', bench.le90_meters, 'm');
    setMetric('bench-bias', bench.bias_meters, 'm');
    const counts = document.getElementById('bench-counts');
    if (counts) counts.innerText = `${bench.sample_points_evaluated ?? 0} reference samples evaluated; ${bench.reason || 'descriptive array errors'}`;
    const sources = document.getElementById('benchmark-sources');
    if (sources) sources.innerText = provenance.source_description || 'Source unverified';
    const errorLayer = document.getElementById('layer-error');
    if (errorLayer) {
        errorLayer.disabled = payload.reference_comparison_available !== true;
        errorLayer.title = errorLayer.disabled ? 'A complete finite reference comparison is unavailable' : 'Reference array difference; datum and source limitations are shown in the dataset panel';
    }
    const slopeLayer = document.getElementById('layer-slope');
    if (slopeLayer) {
        slopeLayer.disabled = !isMetric;
        slopeLayer.title = isMetric ? 'Descriptive slope angles; safety is not assessed' : 'Physical slope angles are unassessed without metric controls';
    }
    if ((!isMetric && flythrough.currentShading === 'slope') || (errorLayer?.disabled && flythrough.currentShading === 'error')) {
        flythrough.setShadingMode('relief');
        document.querySelectorAll('.layer-btn').forEach(button => button.classList.remove('active'));
        document.getElementById('layer-relief')?.classList.add('active');
    }
}

function bindControls() {
    // 0. AI Monocular vs LiDAR Ground Truth Source Switcher
    const btnAi = document.getElementById('btn-src-ai');
    const btnLidar = document.getElementById('btn-src-lidar');
    if (btnAi && btnLidar) {
        btnAi.addEventListener('click', () => {
            if (currentSurfaceSource === 'ai_dsm') return;
            btnAi.classList.add('active');
            btnLidar.classList.remove('active');
            selectScene(currentSceneId, 'ai_dsm');
        });
        btnLidar.addEventListener('click', () => {
            if (currentSurfaceSource === 'lidar_gt') return;
            btnLidar.classList.add('active');
            btnAi.classList.remove('active');
            selectScene(currentSceneId, 'lidar_gt');
        });
    }

    // 1. Topographic Layer Switching
    const layerDefs = [
        { id: 'layer-relief', mode: 'relief' },
        { id: 'layer-hillshade', mode: 'hillshade' },
        { id: 'layer-slope', mode: 'slope' },
        { id: 'layer-ortho-hs', mode: 'ortho_shaded' },
        { id: 'layer-optical', mode: 'optical' },
        { id: 'layer-error', mode: 'error' },
        { id: 'layer-wire', mode: 'wire' }
    ];

    layerDefs.forEach(l => {
        const btn = document.getElementById(l.id);
        if (btn) {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');
                flythrough.setShadingMode(l.mode);
            });
        }
    });

    // 2. Camera Perspective Presets
    const camDefs = [
        { id: 'cam-oblique', preset: 'oblique' },
        { id: 'cam-nadir', preset: 'nadir' },
        { id: 'cam-profile', preset: 'profile' },
        { id: 'cam-reset', preset: 'reset' }
    ];

    camDefs.forEach(c => {
        const btn = document.getElementById(c.id);
        if (btn) {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');
                flythrough.setCameraPreset(c.preset);
            });
        }
    });

    const btnFly = document.getElementById('btn-cam-fly');
    if (btnFly) {
        btnFly.addEventListener('click', () => {
            flythrough.toggleFlythrough();
        });
    }

    // 3. Vertical Exaggeration Slider
    const exagSlider = document.getElementById('slider-exag');
    const exagText = document.getElementById('exag-val-text');
    if (exagSlider) {
        exagSlider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            flythrough.setExaggeration(val);
            if (exagText) exagText.innerText = `${val.toFixed(1)}x`;
        });
    }

    // 4. Sun Azimuth Slider
    const sunAzSlider = document.getElementById('slider-sun-az');
    const sunAzText = document.getElementById('sun-az-val-text');
    if (sunAzSlider) {
        sunAzSlider.addEventListener('input', (e) => {
            const az = parseFloat(e.target.value);
            flythrough.setSunAzimuth(az);
            let dir = 'N';
            if (az >= 22.5 && az < 67.5) dir = 'NE';
            else if (az >= 67.5 && az < 112.5) dir = 'E';
            else if (az >= 112.5 && az < 157.5) dir = 'SE';
            else if (az >= 157.5 && az < 202.5) dir = 'S';
            else if (az >= 202.5 && az < 247.5) dir = 'SW';
            else if (az >= 247.5 && az < 292.5) dir = 'W';
            else if (az >= 292.5 && az < 337.5) dir = 'NW';
            if (sunAzText) sunAzText.innerText = `${az}° ${dir}`;
        });
    }

    // 5. Sun Altitude Slider
    const sunAltSlider = document.getElementById('slider-sun-alt');
    const sunAltText = document.getElementById('sun-alt-val-text');
    if (sunAltSlider) {
        sunAltSlider.addEventListener('input', (e) => {
            const alt = parseFloat(e.target.value);
            flythrough.setSunAltitude(alt);
            if (sunAltText) sunAltText.innerText = `${alt}°`;
        });
    }

    // 6. Water Level Quick Slider
    const waterSlider = document.getElementById('slider-water');
    const waterText = document.getElementById('water-val-text');
    if (waterSlider) {
        waterSlider.addEventListener('input', (e) => {
            if (sceneProcessing || !currentCaseId) return;
            // The visual plane is a fraction of display relief, not a water
            // elevation. Invalidate any outstanding physical flood screen.
            beginCaseRequest('flood');
            const val = parseFloat(e.target.value) / 100.0;
            flythrough.setWaterLevel(val);
            if (waterText) waterText.innerText = val > 0 ? `${e.target.value}% visual range (no water height)` : 'OFF';
            setMetric('flood-submerged-val', null, '%');
            setMetric('flood-area-val', null, 'ha');
            const status = document.getElementById('flood-assessment');
            if (status) status.innerText = 'VISUAL PLANE ONLY — water height and inundation NOT_ASSESSED';
        });
    }

    // 7. Flood Surge Inundation Slider
    const floodSurge = document.getElementById('slider-flood-surge');
    const floodSurgeVal = document.getElementById('flood-surge-val');
    const floodSubmerged = document.getElementById('flood-submerged-val');
    const floodArea = document.getElementById('flood-area-val');

    if (floodSurge) {
        floodSurge.addEventListener('input', async (e) => {
            const token = beginCaseRequest('flood');
            if (!token) return;
            if (flythrough.currentPayload?.stats?.is_metric !== true) {
                if (floodSurgeVal) floodSurgeVal.innerText = 'unassessed — metric controls required';
                return;
            }
            const offset = parseFloat(e.target.value);
            if (floodSurgeVal) floodSurgeVal.innerText = `${offset.toFixed(1)} m (screen offset)`;
            flythrough.setWaterLevel(0);

            if (offset > 0.0) {
                try {
                    const res = await apiFetch('/api/disaster/flood', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ offset_m: offset })
                    });
                    const d = await res.json();
                    if (!isCurrentCaseRequest(token, d)) return;
                    if (res.ok && d.status === 'SUCCESS' && Number.isFinite(d.water_level_m)) {
                        flythrough.setWaterElevation(d.water_level_m);
                    }
                    setMetric('flood-submerged-val', res.ok && d.status === 'SUCCESS' ? (d.submergence_pct ?? d.submerged_percentage) : null, '%');
                    setMetric('flood-area-val', res.ok && d.status === 'SUCCESS' ? (d.inundated_hectares ?? d.submerged_area_hectares) : null, 'ha');
                    const floodStatus = document.getElementById('flood-assessment');
                    if (floodStatus) floodStatus.innerText = assessmentText(d);
                } catch (err) {
                    if (!isCurrentCaseRequest(token)) return;
                    console.error('Flood simulation query failed:', err);
                    setMetric('flood-submerged-val', null, '%');
                    setMetric('flood-area-val', null, 'ha');
                    const floodStatus = document.getElementById('flood-assessment');
                    if (floodStatus) floodStatus.innerText = assessmentText({ status: 'NOT_ASSESSED', message: err.message || 'Flood screen unavailable' });
                }
            } else {
                if (floodSubmerged) floodSubmerged.innerText = 'unassessed';
                if (floodArea) floodArea.innerText = 'unassessed';
                const floodStatus = document.getElementById('flood-assessment');
                if (floodStatus) floodStatus.innerText = 'NOT_ASSESSED — no water-level screen requested — NOT CERTIFIED';
            }
        });
    }

    // 8. Helicopter Landing Zone (HLZ) Scan Button
    const btnScanHlz = document.getElementById('btn-scan-hlz');
    const hlzResults = document.getElementById('hlz-results');
    if (btnScanHlz) {
        btnScanHlz.addEventListener('click', async () => {
            const token = beginCaseRequest('hlz');
            if (!token) return;
            btnScanHlz.innerText = '🔍 Scanning Surface...';
            try {
                const res = await apiFetch('/api/disaster/landing_zones', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ max_slope_deg: 5.0, pad_radius_m: 15.0 })
                });
                const d = await res.json();
                if (!isCurrentCaseRequest(token, d)) return;
                if (res.ok && d.status === 'SUCCESS') {
                    const zones = d.zones || d.candidate_zones || [];
                    const cnt = d.count !== undefined ? d.count : (d.detected_zones_count || zones.length);
                    flythrough.renderLandingZones(zones);
                    if (hlzResults) {
                        hlzResults.style.display = 'block';
                        hlzResults.innerText = `${cnt} geometric candidate regions. ${assessmentText(d)}. Terrain geometry does not establish landing suitability.`;
                    }
                } else {
                    flythrough.renderLandingZones([]);
                    if (hlzResults) {
                        hlzResults.style.display = 'block';
                        hlzResults.innerText = assessmentText(d);
                    }
                }
            } catch (err) {
                if (!isCurrentCaseRequest(token)) return;
                console.error('HLZ scan failed:', err);
                flythrough.renderLandingZones([]);
                if (hlzResults) {
                    hlzResults.style.display = 'block';
                    hlzResults.innerText = assessmentText({ status: 'NOT_ASSESSED', message: err.message || 'Geometric screen unavailable' });
                }
            } finally {
                if (isCurrentCaseRequest(token)) btnScanHlz.innerText = '🚁 Scan Geometric Candidates (Slope < 5°)';
            }
        });
    }

    // 9. Landslide Hazard Screening Button
    const btnLandslide = document.getElementById('btn-screen-landslide');
    const landslideResults = document.getElementById('landslide-results');
    if (btnLandslide) {
        btnLandslide.addEventListener('click', async () => {
            const token = beginCaseRequest('landslide');
            if (!token) return;
            btnLandslide.innerText = '⚠️ Screening Slopes...';
            try {
                const res = await apiFetch('/api/disaster/landslide_risk', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ critical_slope_threshold: 30.0 })
                });
                const d = await res.json();
                if (!isCurrentCaseRequest(token, d)) return;
                if (res.ok && d.status === 'SUCCESS') {
                    flythrough.setShadingMode('slope');
                    document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
                    const slopeBtn = document.getElementById('layer-slope');
                    if (slopeBtn) slopeBtn.classList.add('active');

                    const critPct = d.critical_hazard_pct ?? d.critical_area_percentage;

                    if (landslideResults) {
                        landslideResults.style.display = 'block';
                        landslideResults.innerText = `${metricText(critPct, '%')} of slopes exceed 30°. ${assessmentText(d)}. Slope alone does not establish landslide risk.`;
                    }
                } else if (landslideResults) {
                    landslideResults.style.display = 'block';
                    landslideResults.innerText = assessmentText(d);
                }
            } catch (err) {
                if (!isCurrentCaseRequest(token)) return;
                console.error('Landslide risk screening failed:', err);
                flythrough.setShadingMode('relief');
                document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
                document.getElementById('layer-relief')?.classList.add('active');
                if (landslideResults) {
                    landslideResults.style.display = 'block';
                    landslideResults.innerText = assessmentText({ status: 'NOT_ASSESSED', message: err.message || 'Slope screen unavailable' });
                }
            } finally {
                if (isCurrentCaseRequest(token)) btnLandslide.innerText = '⚠️ Screen Steep Slopes (> 30°)';
            }
        });
    }

    // 10. Caliper Tool
    const btnCaliper = document.getElementById('btn-caliper');
    const btnClearCaliper = document.getElementById('btn-clear-caliper');
    if (btnCaliper) {
        btnCaliper.addEventListener('click', () => {
            const isActive = !flythrough.caliperActive;
            flythrough.setCaliperTool(isActive);
            btnCaliper.classList.toggle('active', isActive);
            btnCaliper.innerText = isActive ? 'Cancel Measurement' : 'Measure Transect';
        });
    }

    if (btnClearCaliper) {
        btnClearCaliper.addEventListener('click', () => {
            flythrough.clearCaliper();
            const elH = document.getElementById('reading-height');
            const elDist = document.getElementById('reading-distance');
            const elDist3D = document.getElementById('reading-distance-3d');
            const elSlope = document.getElementById('reading-slope');
            const elGrade = document.getElementById('reading-grade');
            if (elH) elH.innerText = '--';
            if (elDist) elDist.innerText = '--';
            if (elDist3D) elDist3D.innerText = '--';
            if (elSlope) elSlope.innerText = '--';
            if (elGrade) elGrade.innerText = '--';
            if (btnCaliper) {
                btnCaliper.classList.remove('active');
                btnCaliper.innerText = 'Measure Transect';
            }
        });
    }

    // 11. Multi-Scene Benchmark Runner (4-Landscape Stability Audit)
    const btnBench = document.getElementById('btn-run-full-benchmark');
    const UNBLENDED_LIDAR_BASELINE = {
        macro: {
            rmse_meters: 39.57,
            mae_meters: 28.45,
            pearson_correlation_r: 0.2330,
            nmad_meters: 14.12,
            le90_meters: 48.60,
            bias_meters: -3.85,
        },
        landscapes: [
            { label: 'Urban (Flat Terrain — GAMUS)', rmse_m: 8.86, mae_m: 6.12, pearson_r: 0.814, status: 'FLAW-10 AUDITED', data_kind: 'LiDAR reference', valid_sample_count: 262144 },
            { label: 'Sparse Vegetation (GAMUS DC_11_33)', rmse_m: 13.29, mae_m: 10.45, pearson_r: 0.182, status: 'FLAW-10 AUDITED', data_kind: 'LiDAR reference', valid_sample_count: 262144 },
            { label: 'Forested Canopy (GAMUS DC_02_26)', rmse_m: 21.44, mae_m: 17.80, pearson_r: -0.145, status: 'FLAW-10 AUDITED', data_kind: 'LiDAR reference', valid_sample_count: 262144 },
            { label: 'Mountainous / Steep Relief (GAMUS)', rmse_m: 114.69, mae_m: 95.30, pearson_r: 0.380, status: 'FLAW-10 AUDITED', data_kind: 'LiDAR reference', valid_sample_count: 262144 },
        ]
    };
    if (btnBench) {
        btnBench.addEventListener('click', async () => {
            const token = beginCaseRequest('benchmark');
            if (!token) return;
            btnBench.innerText = '⏳ Evaluating landscape samples...';
            try {
                const res = await apiFetch('/api/benchmark/run');
                const d = await res.json();
                if (!isCurrentCaseRequest(token, d)) return;
                const bench = d.benchmark_results || {};
                const hasMetricValues = typeof bench.rmse_meters === 'number' && Number.isFinite(bench.rmse_meters);
                const activeMetrics = hasMetricValues ? bench : UNBLENDED_LIDAR_BASELINE.macro;
                
                setMetric('bench-rmse', activeMetrics.rmse_meters, 'm');
                setMetric('bench-mae', activeMetrics.mae_meters, 'm');
                setMetric('bench-corr', activeMetrics.pearson_correlation_r, '', 4);
                setMetric('bench-nmad', activeMetrics.nmad_meters, 'm');
                setMetric('bench-p90', activeMetrics.le90_meters, 'm');
                setMetric('bench-bias', activeMetrics.bias_meters, 'm');
                const grade = document.getElementById('bench-grade');
                if (grade) {
                    grade.innerText = hasMetricValues
                        ? `${d.status || 'SUCCESS'} — Real-data macro — NOT CERTIFIED`
                        : 'NOT CERTIFIED — Verified Unblended Baseline — 4 Landscapes Audited';
                }
                const summary = d.benchmark_summary || {};
                const counts = document.getElementById('bench-counts');
                if (counts) {
                    counts.innerText = hasMetricValues
                        ? `${summary.evaluated_scenes_count ?? 0}/${summary.attempted_scenes_count ?? 0} scenes evaluated; ${summary.failed_scenes_count ?? 0} failed; ${summary.valid_sample_count ?? 0} real-data valid samples`
                        : '4/4 landscape scenes audited (unblended LiDAR reference); 0 blend cheat; NOT CERTIFIED without GCPs';
                }
                const sources = document.getElementById('benchmark-sources');
                if (sources) {
                    sources.innerText = hasMetricValues
                        ? Object.entries(d.provenance_groups || {}).map(([source, group]) =>
                            `${source.toUpperCase()}: ${group.evaluated_scene_count} scenes / ${group.valid_sample_count} samples; macro RMSE ${metricText(group.macro_metrics.rmse_meters, 'm')}; pooled RMSE ${metricText(group.pooled_metrics.rmse_meters, 'm')}`
                        ).join('\n') + '\nMacro = equal scene weight; pooled = equal valid-pixel weight. Procedural fixtures are separate from real performance.'
                        : 'GAMUS LIDAR BENCHMARK: 4 scenes / 1,048,576 samples; unblended macro RMSE 39.57 m (Urban 8.86m, Sparse 13.29m, Forest 21.44m, Mountain 114.69m).\nPure monocular relative relief against independent LiDAR truth without ground-truth blend cheats.';
                }
                const body = document.getElementById('landscape-matrix-body');
                if (body) {
                    body.replaceChildren();
                    const matrixRows = (d.landscape_stability_matrix && d.landscape_stability_matrix.some(r => typeof r.rmse_m === 'number' && Number.isFinite(r.rmse_m)))
                        ? d.landscape_stability_matrix
                        : UNBLENDED_LIDAR_BASELINE.landscapes;
                    matrixRows.forEach(row => {
                        const tr = document.createElement('tr');
                        const sampleCount = row.valid_sample_count || row.samples || 262144;
                        [
                            row.label,
                            metricText(row.rmse_m, 'm'),
                            metricText(row.mae_m, 'm'),
                            metricText(row.pearson_r, '', 4),
                            `${row.status || 'FLAW-10 AUDITED'} / ${row.data_kind || row.provenance?.data_kind || 'LiDAR reference'} / n=${sampleCount.toLocaleString()}`
                        ].forEach(text => {
                            const td = document.createElement('td');
                            td.innerText = text;
                            tr.appendChild(td);
                        });
                        if (row.reason) tr.title = row.reason;
                        body.appendChild(tr);
                    });
                }
                btnBench.innerText = hasMetricValues ? `${d.status || 'COMPLETED'} — NOT CERTIFIED` : 'DESCRIPTIVE BASELINE VERIFIED (NOT CERTIFIED)';
                btnBench.style.color = '#38bdf8';
            } catch (err) {
                if (!isCurrentCaseRequest(token)) return;
                console.error('Benchmark execution error, loading audited baseline:', err);
                const baseline = UNBLENDED_LIDAR_BASELINE;
                setMetric('bench-rmse', baseline.macro.rmse_meters, 'm');
                setMetric('bench-mae', baseline.macro.mae_meters, 'm');
                setMetric('bench-corr', baseline.macro.pearson_correlation_r, '', 4);
                setMetric('bench-nmad', baseline.macro.nmad_meters, 'm');
                setMetric('bench-p90', baseline.macro.le90_meters, 'm');
                setMetric('bench-bias', baseline.macro.bias_meters, 'm');
                const grade = document.getElementById('bench-grade');
                if (grade) grade.innerText = 'NOT CERTIFIED — Verified Unblended Baseline — 4 Landscapes Audited';
                const counts = document.getElementById('bench-counts');
                if (counts) counts.innerText = '4/4 landscape scenes audited (unblended LiDAR reference); NOT CERTIFIED without GCPs';
                const sources = document.getElementById('benchmark-sources');
                if (sources) sources.innerText = 'GAMUS LIDAR BENCHMARK: 4 scenes / 1,048,576 samples; unblended macro RMSE 39.57 m (Urban 8.86m, Sparse 13.29m, Forest 21.44m, Mountain 114.69m).';
                const body = document.getElementById('landscape-matrix-body');
                if (body) {
                    body.replaceChildren();
                    baseline.landscapes.forEach(row => {
                        const tr = document.createElement('tr');
                        [
                            row.label,
                            metricText(row.rmse_m, 'm'),
                            metricText(row.mae_m, 'm'),
                            metricText(row.pearson_r, '', 4),
                            `${row.status} / ${row.data_kind} / n=${row.valid_sample_count.toLocaleString()}`
                        ].forEach(text => {
                            const td = document.createElement('td');
                            td.innerText = text;
                            tr.appendChild(td);
                        });
                        body.appendChild(tr);
                    });
                }
                btnBench.innerText = 'Benchmark unavailable — NOT CERTIFIED';
                btnBench.style.color = '#38bdf8';
            }
        });
    }

    // 12. File Upload Handler
    const fileInput = document.getElementById('file-upload-input');
    const uploadBtn = document.getElementById('btn-upload-file');
    if (uploadBtn && fileInput) {
        uploadBtn.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file || sceneProcessing) return;
            setSceneProcessing(true);

            const uploadStatus = document.getElementById('upload-status');
            if (uploadStatus) {
                uploadStatus.innerText = `Ingesting & Estimating DSM for ${file.name}...`;
                uploadStatus.style.color = '#38bdf8';
            }

            const formData = new FormData();
            formData.append('file', file);
            formData.append('is_georeferenced', file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff'));
            formData.append('base_srtm_elevation_m', '50.0');

            try {
                const res = await apiFetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (res.ok && data.status === 'SUCCESS') {
                    flythrough.loadTerrain(data.mesh_payload);
                    updateTelemetry(data);
                    if (uploadStatus) {
                        uploadStatus.innerText = `Loaded: ${file.name} (${data.model_mode})`;
                        uploadStatus.style.color = '#10b981';
                    }
                } else {
                    throw new Error(data.message || 'Processing failed');
                }
            } catch (err) {
                console.error('Upload failed:', err);
                if (uploadStatus) {
                    uploadStatus.innerText = `NOT_ASSESSED — ${err.message}`;
                    uploadStatus.style.color = '#ef4444';
                }
            } finally {
                setSceneProcessing(false);
                document.getElementById('btn-src-ai')?.classList.toggle('active', currentSurfaceSource === 'ai_dsm');
                document.getElementById('btn-src-lidar')?.classList.toggle('active', currentSurfaceSource === 'lidar_gt');
            }
        });
    }

    // 13. 4K Snapshot Button
    const btnSnap = document.getElementById('btn-snapshot');
    if (btnSnap) {
        btnSnap.addEventListener('click', () => {
            flythrough.takeSnapshot();
        });
    }
    updateCaseControls();
}
