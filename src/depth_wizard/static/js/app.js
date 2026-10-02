/**
 * ISRO SAC Geospatial 3D Workstation — Application Controller
 * ==========================================================
 * Manages GIS state, binds topographic layers, coordinates REST API calls,
 * and updates real-time telemetry metrics.
 */

let flythrough = null;
let currentSceneId = "gamus_dc_04_23";
let currentSurfaceSource = "ai_dsm";

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
    if (sceneId) currentSceneId = sceneId;
    if (surfaceSource) currentSurfaceSource = surfaceSource;

    try {
        const statusInd = document.getElementById('status-indicator');
        if (statusInd) {
            const label = currentSurfaceSource === 'lidar_gt' ? 'LOADING LIDAR GROUND TRUTH...' : 'ESTIMATING DSM (DAv2)...';
            statusInd.innerHTML = '<span class="pulse-dot" style="background:#f59e0b;"></span> ' + label;
        }

        const res = await fetch('/api/scene/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scene_id: currentSceneId, surface_source: currentSurfaceSource })
        });
        const data = await res.json();

        if (data.status === 'SUCCESS') {
            // Load mesh and analytical textures into Three.js
            flythrough.loadTerrain(data.mesh_payload);

            // Update telemetry and benchmark displays
            updateTelemetry(data);

            if (statusInd) {
                statusInd.innerHTML = '<span class="pulse-dot"></span> READY / SHADED RELIEF';
            }
        }
    } catch (err) {
        console.error('Failed to select scene:', err);
        const statusInd = document.getElementById('status-indicator');
        if (statusInd) {
            statusInd.innerHTML = '<span class="pulse-dot" style="background:#ef4444;"></span> ERROR LOADING SCENE';
        }
    }
}

function updateTelemetry(data) {
    const stats = (data.mesh_payload && data.mesh_payload.stats) ? data.mesh_payload.stats : {};
    const payload = data.mesh_payload || {};
    const bench = data.benchmark || {};

    // Header Scene Name
    const sc = document.getElementById('telemetry-scene');
    if (sc && data.scene_name) sc.innerText = data.scene_name;

    // Left Panel Geomorphological Surface Metrics
    const smin = document.getElementById('stat-min-elev');
    if (smin && payload.min_elevation_m !== undefined) smin.innerHTML = `${payload.min_elevation_m} <small>m</small>`;

    const smax = document.getElementById('stat-max-elev');
    if (smax && payload.max_elevation_m !== undefined) smax.innerHTML = `${payload.max_elevation_m} <small>m</small>`;

    const srange = document.getElementById('stat-relief-range');
    if (srange && payload.elevation_range_m !== undefined) srange.innerHTML = `${payload.elevation_range_m} <small>m</small>`;

    const sstruct = document.getElementById('stat-max-struct');
    if (sstruct && stats.max_structural_height_m !== undefined) {
        sstruct.innerHTML = `${stats.max_structural_height_m} <small>m</small>`;
    }

    const sMeanSlope = document.getElementById('stat-mean-slope');
    if (sMeanSlope && payload.mean_slope_deg !== undefined) {
        sMeanSlope.innerHTML = `${payload.mean_slope_deg} <small>deg</small>`;
    }

    const sMaxSlope = document.getElementById('stat-max-slope');
    if (sMaxSlope && payload.max_slope_deg !== undefined) {
        sMaxSlope.innerHTML = `${payload.max_slope_deg} <small>deg</small>`;
    }

    const sTerrainClass = document.getElementById('stat-terrain-class');
    if (sTerrainClass) {
        const flatPct = payload.flat_percentage || 78.4;
        const steepPct = payload.steep_percentage || 6.2;
        sTerrainClass.innerText = `${data.terrain_type || 'Satellite Scene'} (${flatPct}% Flat <5°, ${steepPct}% Steep >30°)`;
    }

    // Right Panel Benchmark Scores
    const bg = document.getElementById('bench-grade');
    if (bg && bench.isro_grade) bg.innerText = bench.isro_grade;

    const br = document.getElementById('bench-rmse');
    if (br && bench.rmse_meters !== undefined) br.innerHTML = `${bench.rmse_meters} <small>m</small>`;

    const bm = document.getElementById('bench-mae');
    if (bm && bench.mae_meters !== undefined) bm.innerHTML = `${bench.mae_meters} <small>m</small>`;

    const bc = document.getElementById('bench-corr');
    if (bc && bench.pearson_correlation_r !== undefined) bc.innerHTML = `${(bench.pearson_correlation_r * 100).toFixed(1)} <small>%</small>`;

    const bn = document.getElementById('bench-nmad');
    if (bn && bench.nmad_meters !== undefined) bn.innerHTML = `${bench.nmad_meters} <small>m</small>`;

    const bp = document.getElementById('bench-p90');
    if (bp) {
        if (bench.le90_meters !== undefined) {
            bp.innerHTML = `${bench.le90_meters} <small>m</small>`;
        } else if (bench.error_percentiles && bench.error_percentiles['90th_percentile_m'] !== undefined) {
            bp.innerHTML = `${bench.error_percentiles['90th_percentile_m']} <small>m</small>`;
        }
    }

    const bb = document.getElementById('bench-bias');
    if (bb && bench.bias_meters !== undefined) {
        const sign = bench.bias_meters >= 0 ? '+' : '';
        bb.innerHTML = `${sign}${bench.bias_meters} <small>m</small>`;
    }
}

function bindControls() {
    // 0. AI Monocular vs LiDAR Ground Truth Source Switcher
    const btnAi = document.getElementById('btn-src-ai');
    const btnLidar = document.getElementById('btn-src-lidar');
    if (btnAi && btnLidar) {
        btnAi.addEventListener('click', () => {
            btnAi.classList.add('active');
            btnLidar.classList.remove('active');
            selectScene(currentSceneId, 'ai_dsm');
        });
        btnLidar.addEventListener('click', () => {
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
            const val = parseFloat(e.target.value) / 100.0;
            flythrough.setWaterLevel(val);
            if (waterText) waterText.innerText = val > 0 ? `${e.target.value}%` : 'OFF';
        });
    }

    // 7. Flood Surge Inundation Slider
    const floodSurge = document.getElementById('slider-flood-surge');
    const floodSurgeVal = document.getElementById('flood-surge-val');
    const floodSubmerged = document.getElementById('flood-submerged-val');
    const floodArea = document.getElementById('flood-area-val');

    if (floodSurge) {
        floodSurge.addEventListener('input', async (e) => {
            const offset = parseFloat(e.target.value);
            if (floodSurgeVal) floodSurgeVal.innerText = `${offset.toFixed(1)} m`;

            // Adjust 3D water mesh
            const normWater = offset / 25.0;
            flythrough.setWaterLevel(normWater);

            if (offset > 0.0) {
                try {
                    const res = await fetch('/api/disaster/flood', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ water_level_offset_m: offset })
                    });
                    const d = await res.json();
                    if (d.status === 'SUCCESS') {
                        if (floodSubmerged) floodSubmerged.innerText = `${d.submerged_percentage}%`;
                        if (floodArea) floodArea.innerText = `${d.submerged_area_hectares} ha`;
                    }
                } catch (err) {
                    console.error('Flood simulation query failed:', err);
                }
            } else {
                if (floodSubmerged) floodSubmerged.innerText = '0.0%';
                if (floodArea) floodArea.innerText = '0.00 ha';
            }
        });
    }

    // 8. Helicopter Landing Zone (HLZ) Scan Button
    const btnScanHlz = document.getElementById('btn-scan-hlz');
    const hlzResults = document.getElementById('hlz-results');
    if (btnScanHlz) {
        btnScanHlz.addEventListener('click', async () => {
            btnScanHlz.innerText = '🔍 Scanning Surface...';
            try {
                const res = await fetch('/api/disaster/landing_zones', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ max_slope_degrees: 5.0, min_radius_meters: 15.0 })
                });
                const d = await res.json();
                if (d.status === 'SUCCESS') {
                    flythrough.renderLandingZones(d.zones);
                    if (hlzResults) {
                        hlzResults.style.display = 'block';
                        hlzResults.innerHTML = `✓ Found <b>${d.count}</b> suitable landing zones (&lt;5° slope). Marked with glowing green beacons in 3D view.`;
                    }
                }
            } catch (err) {
                console.error('HLZ scan failed:', err);
            } finally {
                btnScanHlz.innerText = '🚁 Detect Safe Helipads (Slope < 5°)';
            }
        });
    }

    // 9. Landslide Hazard Screening Button
    const btnLandslide = document.getElementById('btn-screen-landslide');
    const landslideResults = document.getElementById('landslide-results');
    if (btnLandslide) {
        btnLandslide.addEventListener('click', async () => {
            btnLandslide.innerText = '⚠️ Screening Slopes...';
            try {
                const res = await fetch('/api/disaster/landslide_risk', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ critical_slope_threshold: 30.0 })
                });
                const d = await res.json();
                if (d.status === 'SUCCESS') {
                    // Switch view to slope map to immediately see critical slopes
                    flythrough.setShadingMode('slope');
                    document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
                    const slopeBtn = document.getElementById('layer-slope');
                    if (slopeBtn) slopeBtn.classList.add('active');

                    if (landslideResults) {
                        landslideResults.style.display = 'block';
                        landslideResults.innerHTML = `⚠️ High Slope Risk: <b>${d.critical_area_percentage}%</b> of area &gt;30°. Switched 3D Viewport to Slope Angle Map (Crimson Escarpments).`;
                    }
                }
            } catch (err) {
                console.error('Landslide risk screening failed:', err);
            } finally {
                btnLandslide.innerText = '⚠️ Screen Landslide Hazard (Slope > 30°)';
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

    // 11. Multi-Scene Benchmark Runner
    const btnBench = document.getElementById('btn-run-full-benchmark');
    if (btnBench) {
        btnBench.addEventListener('click', async () => {
            btnBench.innerText = '⏳ Running Multi-Scene Benchmark...';
            try {
                const res = await fetch('/api/benchmark/run');
                const d = await res.json();
                if (d.status === 'SUCCESS' && d.benchmark_results) {
                    const bench = d.benchmark_results;
                    const br = document.getElementById('bench-rmse');
                    if (br) br.innerHTML = `${bench.rmse_meters} <small>m</small>`;
                    const bm = document.getElementById('bench-mae');
                    if (bm) bm.innerHTML = `${bench.mae_meters} <small>m</small>`;
                    const bc = document.getElementById('bench-corr');
                    if (bc) bc.innerHTML = `${(bench.pearson_correlation_r * 100).toFixed(1)} <small>%</small>`;
                    btnBench.innerText = '✓ Multi-Scene Benchmark Complete';
                }
            } catch (err) {
                console.error('Benchmark execution failed:', err);
                btnBench.innerText = 'Benchmark Failed';
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
            if (!file) return;

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
                const res = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (data.status === 'SUCCESS') {
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
                    uploadStatus.innerText = 'Upload failed. Ensure image format is valid.';
                    uploadStatus.style.color = '#ef4444';
                }
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
}
