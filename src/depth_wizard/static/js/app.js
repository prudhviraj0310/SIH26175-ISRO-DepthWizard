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
    const stats = data.mesh_payload.stats;
    const bench = data.benchmark;

    // Header Telemetry
    document.getElementById('telemetry-scene').innerText = data.scene_name;
    document.getElementById('telemetry-terrain').innerText = data.terrain_type;

    // Left Panel Stats
    document.getElementById('stat-base-srtm').innerText = stats.base_srtm_m + ' m';
    document.getElementById('stat-min-elev').innerText = stats.min_elevation_m + ' m';
    document.getElementById('stat-max-elev').innerText = stats.max_elevation_m + ' m';
    document.getElementById('stat-max-struct').innerText = stats.max_structural_height_m + ' m';
    document.getElementById('stat-mean-struct').innerText = stats.mean_structural_height_m + ' m';

    // Right Panel Benchmark
    document.getElementById('bench-grade').innerText = bench.isro_grade;
    document.getElementById('bench-rmse').innerText = bench.rmse_meters + ' m';
    document.getElementById('bench-mae').innerText = bench.mae_meters + ' m';
    document.getElementById('bench-corr').innerText = (bench.pearson_correlation_r * 100).toFixed(1) + ' %';
    document.getElementById('bench-p90').innerText = bench.error_percentiles['90th_percentile_m'] + ' m';
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
