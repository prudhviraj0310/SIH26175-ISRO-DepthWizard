/**
 * DepthWizard Mission Control Application Controller
 * ==================================================
 * Liaises with the FastAPI backend, manages cockpit state, and triggers benchmarks.
 */

let flythrough = null;

document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initialize Three.js Engine
    flythrough = new FlythroughEngine('flythrough-canvas-container');

    // 2. Load available scenes
    await loadScenes();

    // 3. Bind UI Controls
    bindControls();

    // 4. Load initial default scene (ISRO SAC Ahmedabad)
    await selectScene('isro_sac_ahmedabad');
});

async function loadScenes() {
    try {
        const res = await fetch('/api/scenes');
        const data = await res.json();
        const selectElem = document.getElementById('scene-selector');
        selectElem.innerHTML = '';
        
        data.scenes.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.innerText = `${s.name} (${s.terrain_type})`;
            selectElem.appendChild(opt);
        });

        selectElem.addEventListener('change', async (e) => {
            await selectScene(e.target.value);
        });
    } catch (err) {
        console.error('Failed to load scenes:', err);
    }
}

async function selectScene(sceneId) {
    try {
        document.getElementById('status-indicator').innerText = 'COMPUTING DSM...';
        document.getElementById('status-indicator').style.color = 'var(--accent-amber)';

        const res = await fetch('/api/scene/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scene_id: sceneId })
        });
        const data = await res.json();

        if (data.status === 'SUCCESS') {
            // Update 3D Terrain
            flythrough.loadTerrain(data.mesh_payload);

            // Update Telemetry Panel
            updateTelemetry(data);

            document.getElementById('status-indicator').innerText = 'OPERATIONAL (3D LIVE)';
            document.getElementById('status-indicator').style.color = 'var(--accent-green)';
        }
    } catch (err) {
        console.error('Failed to select scene:', err);
        document.getElementById('status-indicator').innerText = 'ERROR';
        document.getElementById('status-indicator').style.color = 'var(--accent-crimson)';
    }
}

function updateTelemetry(data) {
    const stats = (data.mesh_payload && data.mesh_payload.stats) ? data.mesh_payload.stats : {};
    const bench = data.benchmark || {};

    // Header Telemetry
    const sc = document.getElementById('telemetry-scene');
    if (sc && data.scene_name) sc.innerText = data.scene_name;
    const tt = document.getElementById('telemetry-terrain');
    if (tt && data.terrain_type) tt.innerText = data.terrain_type;

    // Left Panel Stats
    const sm = document.getElementById('stat-model-mode');
    if (sm) sm.innerText = data.model_mode || stats.model_mode || 'Absolute DSM';
    const sb = document.getElementById('stat-base-srtm');
    if (sb && stats.base_srtm_m !== undefined) sb.innerText = stats.base_srtm_m + ' m';
    const smin = document.getElementById('stat-min-elev');
    if (smin && stats.min_elevation_m !== undefined) smin.innerText = stats.min_elevation_m + ' m';
    const smax = document.getElementById('stat-max-elev');
    if (smax && stats.max_elevation_m !== undefined) smax.innerText = stats.max_elevation_m + ' m';
    const sst = document.getElementById('stat-max-struct');
    if (sst && stats.max_structural_height_m !== undefined) sst.innerText = stats.max_structural_height_m + ' m';

    // Right Panel Benchmark
    const bg = document.getElementById('bench-grade');
    if (bg && bench.isro_grade) bg.innerText = bench.isro_grade;
    const br = document.getElementById('bench-rmse');
    if (br && bench.rmse_meters !== undefined) br.innerText = bench.rmse_meters + ' m';
    const bm = document.getElementById('bench-mae');
    if (bm && bench.mae_meters !== undefined) bm.innerText = bench.mae_meters + ' m';
    const bc = document.getElementById('bench-corr');
    if (bc && bench.pearson_correlation_r !== undefined) bc.innerText = (bench.pearson_correlation_r * 100).toFixed(1) + ' %';
    const bp = document.getElementById('bench-p90');
    if (bp) {
        if (bench.le90_meters !== undefined) {
            bp.innerText = bench.le90_meters + ' m';
        } else if (bench.error_percentiles && bench.error_percentiles['90th_percentile_m'] !== undefined) {
            bp.innerText = bench.error_percentiles['90th_percentile_m'] + ' m';
        }
    }
}

function bindControls() {
    // Mode toggles
    document.getElementById('btn-mode-orbit').addEventListener('click', (e) => {
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        e.currentTarget.classList.add('active');
        flythrough.setMode('orbit');
    });

    document.getElementById('btn-mode-drone').addEventListener('click', (e) => {
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        e.currentTarget.classList.add('active');
        flythrough.setMode('drone');
    });

    // File Upload Handler (PNG / JPG / TIFF)
    const fileInput = document.getElementById('file-upload-input');
    const uploadBtn = document.getElementById('btn-upload-file');
    uploadBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const uploadStatus = document.getElementById('upload-status');
        uploadStatus.innerText = `Processing ${file.name}...`;
        uploadStatus.style.color = 'var(--accent-cyan)';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('is_georeferenced', file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff'));
        formData.append('base_srtm_elevation_m', '50.0');

        try {
            document.getElementById('status-indicator').innerText = 'UPLOADING & ESTIMATING DSM...';
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.status === 'SUCCESS') {
                flythrough.loadTerrain(data.mesh_payload);
                updateTelemetry(data);
                uploadStatus.innerText = `Loaded: ${file.name} (${data.model_mode})`;
                uploadStatus.style.color = 'var(--accent-green)';
                document.getElementById('status-indicator').innerText = 'OPERATIONAL (3D LIVE)';
                document.getElementById('status-indicator').style.color = 'var(--accent-green)';
            } else {
                throw new Error(data.message || 'Processing failed');
            }
        } catch (err) {
            console.error('Upload failed:', err);
            uploadStatus.innerText = 'Upload failed';
            uploadStatus.style.color = 'var(--accent-crimson)';
        }
    });

    // Shading Mode Toggles (RGB, Elevation Tint, Slope Hazard)
    const shadeBtns = [
        { id: 'btn-shade-optical', mode: 'optical' },
        { id: 'btn-shade-elev', mode: 'elevation' },
        { id: 'btn-shade-slope', mode: 'slope' }
    ];
    shadeBtns.forEach(sb => {
        const elem = document.getElementById(sb.id);
        if (elem) {
            elem.addEventListener('click', (e) => {
                shadeBtns.forEach(b => document.getElementById(b.id).classList.remove('active'));
                e.currentTarget.classList.add('active');
                flythrough.setShadingMode(sb.mode);
            });
        }
    });

    // Camera Perspective Presets (Nadir, Oblique, FPV)
    const camBtns = [
        { id: 'btn-cam-nadir', view: 'nadir' },
        { id: 'btn-cam-oblique', view: 'oblique' },
        { id: 'btn-cam-fpv', view: 'fpv' }
    ];
    camBtns.forEach(cb => {
        const elem = document.getElementById(cb.id);
        if (elem) {
            elem.addEventListener('click', (e) => {
                camBtns.forEach(b => document.getElementById(b.id).classList.remove('active'));
                e.currentTarget.classList.add('active');
                flythrough.setCameraView(cb.view);
            });
        }
    });

    // Caliper Laser Tool
    document.getElementById('btn-caliper').addEventListener('click', () => {
        const isActive = !flythrough.caliperActive;
        flythrough.setCaliperTool(isActive);
    });

    document.getElementById('btn-clear-caliper').addEventListener('click', () => {
        flythrough.clearCaliper();
        document.getElementById('reading-height').innerText = '--';
        document.getElementById('reading-distance').innerText = '--';
        document.getElementById('reading-slope').innerText = '--';
        document.getElementById('caliper-status').innerText = 'CLEARED';
    });

    // Flood simulation water slider
    const waterSlider = document.getElementById('slider-water');
    waterSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value) / 100.0;
        flythrough.setWaterLevel(val);
        document.getElementById('water-val-text').innerText = e.target.value + '%';
    });

    // Sun angle slider
    const sunSlider = document.getElementById('slider-sun');
    sunSlider.addEventListener('input', (e) => {
        const angle = parseFloat(e.target.value);
        flythrough.setSunAngle(angle);
        document.getElementById('sun-val-text').innerText = angle + '°';
    });

    // Full Benchmark Run button
    
    // Exaggeration Slider
    const exagSlider = document.getElementById('slider-exag');
    if (exagSlider) {
        exagSlider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            flythrough.setExaggeration(val);
            document.getElementById('exag-val-text').innerText = val.toFixed(1) + 'x';
        });
    }

    // Wireframe Mode
    const wireBtn = document.getElementById('btn-shade-wire');
    if (wireBtn) {
        wireBtn.addEventListener('click', () => {
            const isWire = flythrough.toggleWireframe();
            wireBtn.classList.toggle('active', isWire);
        });
    }

    // 4K Snapshot
    const snapBtn = document.getElementById('btn-snapshot');
    if (snapBtn) {
        snapBtn.addEventListener('click', () => {
            flythrough.takeSnapshot();
        });
    }

    document.getElementById('btn-run-full-benchmark').addEventListener('click', async () => {
        const btn = document.getElementById('btn-run-full-benchmark');
        btn.innerText = 'RUNNING BENCHMARKS...';
        try {
            const res = await fetch('/api/benchmark');
            const data = await res.json();
            alert(`ISRO SAC Multi-Scene Benchmark Report:\n\n` +
                  `Overall Compliance: ${data.benchmark_summary.overall_isro_compliance}\n` +
                  `Average RMSE: ${data.benchmark_summary.average_rmse_meters} meters\n` +
                  `Average Pearson Correlation: ${(data.benchmark_summary.average_correlation_r * 100).toFixed(1)}%\n` +
                  `Evaluated Across 3 Landscape Archetypes.`);
            btn.innerText = 'RUN ISRO BENCHMARK SUITE';
        } catch (err) {
            console.error(err);
            btn.innerText = 'BENCHMARK FAILED';
        }
    });
}
