/* Offline runtime regressions: real controllers, deferred fetches, minimal DOM. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const root = path.resolve(__dirname, '..');
const staticDir = path.join(root, 'src/depth_wizard/static/js');

class Element {
    constructor(id) {
        this.id = id;
        this.innerText = '';
        this.style = {};
        this.value = '0';
        this.disabled = false;
        this.children = [];
        this.listeners = new Map();
        this.attributes = {};
        const classes = new Set();
        this.classList = {
            add: name => classes.add(name), remove: name => classes.delete(name),
            toggle: (name, on) => on ? classes.add(name) : classes.delete(name),
        };
    }
    set innerHTML(value) { this.html = value; this.children = []; }
    get innerHTML() { return this.html || ''; }
    addEventListener(type, callback) { this.listeners.set(type, callback); }
    setAttribute(name, value) { this.attributes[name] = value; }
    appendChild(child) { this.children.push(child); }
    replaceChildren(...children) { this.children = children; this.html = ''; }
    fire(type, value) {
        if (value !== undefined) this.value = String(value);
        return this.listeners.get(type)?.({ target: this, currentTarget: this });
    }
}

function harness() {
    const html = fs.readFileSync(path.join(root, 'src/depth_wizard/templates/index.html'), 'utf8');
    const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(match => [match[1], new Element(match[1])]));
    const document = {
        getElementById: id => elements.get(id) || null,
        querySelectorAll: () => [...elements.values()].filter(element => element.id.startsWith('layer-')),
        createElement: tag => new Element(tag),
        addEventListener() {},
    };
    const requests = [];
    const fly = {
        currentPayload: null, currentShading: 'relief', zones: [], levels: [], elevations: [],
        loadTerrain(payload) { this.currentPayload = payload; },
        clearLandingZones() { this.zones = []; }, clearCaliper() {},
        setCaliperTool(active) { this.caliperActive = active; },
        setWaterLevel(level) { this.levels.push(level); },
        setWaterElevation(level) { this.elevations.push(level); },
        renderLandingZones(zones) { this.zones = zones; },
        setShadingMode(mode) { this.currentShading = mode; },
    };
    const context = vm.createContext({
        document, Headers, FormData, stubFly: fly, console: { error() {}, log() {} },
        fetch(url, options) {
            return new Promise((resolve, reject) => requests.push({ url, options, resolve, reject }));
        },
    });
    const run = code => vm.runInContext(code, context);
    run(fs.readFileSync(path.join(staticDir, 'app.js'), 'utf8'));
    run('flythrough = stubFly; bindControls();');
    function publish(id, metric = true, lod1 = {}) {
        const data = {
            case_id: id, scene_name: id, surface_source: 'ai_dsm', terrain_type: 'Controlled UI fixture',
            provenance: { data_kind: 'synthetic', source_description: 'Runtime test fixture' },
            benchmark: { status: 'NOT_ASSESSED', rmse_meters: null },
            mesh_payload: {
                stats: { is_metric: metric, gsd_m: 1, grid_dimensions: [40, 40] },
                min_elevation_m: 50, max_elevation_m: 60, elevation_range_m: 10,
                lod1_status: 'EMPTY', lod1_building_count: 0, lod1_dtm_source: 'TEST_TERRAIN',
                lod1_dtm_status: 'SUPPLIED_UNVERIFIED_REFERENCE', lod1_dtm_authoritative: false,
                ...lod1,
            },
        };
        context.caseData = data;
        run('flythrough.loadTerrain(caseData.mesh_payload); updateTelemetry(caseData); setSceneProcessing(false);');
        return data;
    }
    function respond(request, data, ok = true) {
        request.resolve({ ok, json: async () => data });
    }
    return { elements, document, fly, context, requests, run, publish, respond };
}

const suiteReport = (rmse = 7) => ({
    status: 'SUCCESS', benchmark_results: { rmse_meters: rmse },
    benchmark_summary: { evaluated_scenes_count: 1, attempted_scenes_count: 1, valid_sample_count: 16 },
    provenance_groups: {}, landscape_stability_matrix: [],
});

test('outstanding flood/HLZ/slope/benchmark results cannot populate the newly selected case', async () => {
    const h = harness();
    h.publish('A');
    const flood = h.elements.get('slider-flood-surge').fire('input', 3);
    const hlz = h.elements.get('btn-scan-hlz').fire('click');
    const slope = h.elements.get('btn-screen-landslide').fire('click');
    const benchmark = h.elements.get('btn-run-full-benchmark').fire('click');
    const old = h.requests.slice();
    const selecting = h.run("selectScene('scene-B')");
    assert.equal(h.elements.get('btn-scan-hlz').disabled, true);
    assert.equal(h.elements.get('btn-run-full-benchmark').disabled, true);
    assert.equal(h.elements.get('slider-flood-surge').disabled, true);
    await h.elements.get('btn-scan-hlz').fire('click');
    assert.equal(h.requests.length, 5, 'Loading controls must not dispatch another request');
    const data = {
        ...h.context.caseData, case_id: 'B', scene_name: 'B',
        mesh_payload: { ...h.context.caseData.mesh_payload, stats: { is_metric: false } },
    };
    h.respond(h.requests[4], { ...data, status: 'SUCCESS' });
    await selecting;
    h.respond(old[0], { status: 'SUCCESS', case_id: 'A', water_level_m: 53, submergence_pct: 99, inundated_hectares: 10 });
    h.respond(old[1], { status: 'SUCCESS', case_id: 'A', zones: [{ norm_x: 0.5, norm_y: 0.5 }], count: 1 });
    h.respond(old[2], { status: 'SUCCESS', case_id: 'A', critical_hazard_pct: 90 });
    h.respond(old[3], suiteReport(99));
    await Promise.all([flood, hlz, slope, benchmark]);
    assert.equal(h.elements.get('flood-submerged-val').innerText, 'unassessed');
    assert.equal(h.fly.zones.length, 0);
    assert.equal(h.fly.currentShading, 'relief');
    assert.equal(h.elements.get('bench-rmse').innerText, 'unassessed');
    assert.match(h.elements.get('source-lod1').innerText, /EMPTY.*TEST_TERRAIN.*SUPPLIED_UNVERIFIED_REFERENCE/);
    assert.equal(h.fly.elevations.length, 0);
});

test('stale network failures and mismatched case receipts cannot clear a current result', async () => {
    const h = harness();
    h.publish('A');
    const old = h.elements.get('btn-run-full-benchmark').fire('click');
    h.publish('B');
    const fresh = h.elements.get('btn-run-full-benchmark').fire('click');
    h.respond(h.requests[1], suiteReport(7));
    await fresh;
    h.requests[0].reject(new Error('stale failure'));
    await old;
    assert.equal(h.elements.get('bench-rmse').innerText, '7.00 m');
    const flood = h.elements.get('slider-flood-surge').fire('input', 3);
    h.respond(h.requests[2], { status: 'SUCCESS', case_id: 'wrong-case', submergence_pct: 88 });
    await flood;
    assert.equal(h.elements.get('flood-submerged-val').innerText, 'unassessed');
});

test('same-case out-of-order flood responses and an OFF slider cannot restore stale values', async () => {
    const h = harness();
    h.publish('A');
    const old = h.elements.get('slider-flood-surge').fire('input', 2);
    const fresh = h.elements.get('slider-flood-surge').fire('input', 4);
    h.respond(h.requests[1], { status: 'SUCCESS', case_id: 'A', water_level_m: 54, submergence_pct: 40 });
    await fresh;
    h.respond(h.requests[0], { status: 'SUCCESS', case_id: 'A', water_level_m: 52, submergence_pct: 20 });
    await old;
    assert.equal(h.elements.get('flood-submerged-val').innerText, '40.00 %');
    assert.deepEqual(h.fly.elevations, [54]);
    const pending = h.elements.get('slider-flood-surge').fire('input', 5);
    await h.elements.get('slider-flood-surge').fire('input', 0);
    h.respond(h.requests[2], { status: 'SUCCESS', case_id: 'A', water_level_m: 55, submergence_pct: 50 });
    await pending;
    assert.equal(h.elements.get('flood-submerged-val').innerText, 'unassessed');
    assert.deepEqual(h.fly.elevations, [54]);
});

test('unscaled scenes expose visual-plane semantics and failed LoD-1 is unavailable, not zero blocks', async () => {
    const h = harness();
    h.publish('A', false, { lod1_status: 'FAILED', lod1_error: 'fixture failure' });
    assert.equal(h.elements.get('slider-flood-surge').disabled, true);
    await h.elements.get('slider-flood-surge').fire('input', 4);
    assert.equal(h.requests.length, 0);
    assert.match(h.elements.get('flood-surge-val').innerText, /unassessed/);
    h.elements.get('slider-water').fire('input', 50);
    assert.equal(h.fly.levels.at(-1), 0.5);
    assert.match(h.elements.get('water-val-text').innerText, /50% visual range.*no water height/);
    assert.match(h.elements.get('source-lod1').innerText, /FAILED.*fixture failure/);
    assert.doesNotMatch(h.elements.get('source-lod1').innerText, /0 blocks/);
});

function rendererHarness() {
    const h = harness();
    const three = require(path.join(staticDir, 'three.min.js'));
    const textures = [];
    h.context.THREE = {
        ...three,
        TextureLoader: class { load(url, callback) { textures.push({ url, callback }); } },
    };
    h.run(fs.readFileSync(path.join(staticDir, 'flythrough3d.js'), 'utf8'));
    h.run(`visualizer = Object.assign(Object.create(FlythroughEngine.prototype), {
        scene: new THREE.Scene(), target: new THREE.Vector3(), planeDim: 180, groundDimM: 256,
        currentExaggeration: 1, currentShading: 'relief', waterMesh: { position: { z: -100 } },
        clearCaliper() {}, clearLandingZones() {}, buildPlinthAndSkirt() {}, updateCameraOrbit() {}
    });`);
    return { ...h, textures };
}

test('physical water elevation uses API coordinates and is refused for unscaled surfaces', () => {
    const h = rendererHarness();
    h.run(`visualizer.currentPayload = { stats: { is_metric: false }, min_elevation_m: 50, elevation_range_m: 10 };
        visualizer.setWaterElevation(53);`);
    assert.equal(h.run('visualizer.waterMesh.position.z'), -100);
    h.run('visualizer.currentPayload.stats.is_metric = true; visualizer.setWaterElevation(53);');
    assert.equal(h.run('visualizer.waterMesh.position.z'), 3 * 180 / 256);
    h.run('visualizer.setWaterElevation(Infinity); visualizer.setWaterLevel(NaN);');
    assert.equal(h.run('visualizer.waterMesh.position.z'), 3 * 180 / 256);
});

test('late terrain texture callbacks cannot replace a new case texture', () => {
    const h = rendererHarness();
    const payload = url => ({
        grid_size: 2, normalized_z: [0, 0, 1, 1], min_elevation_m: 50, max_elevation_m: 60,
        elevation_range_m: 10, stats: { is_metric: false }, relief_texture_url: url,
    });
    h.context.a = payload('case-A');
    h.context.b = payload('case-B');
    h.run('visualizer.loadTerrain(a); visualizer.loadTerrain(b);');
    let disposed = false;
    h.textures[0].callback({ dispose() { disposed = true; } });
    assert.equal(disposed, true);
    const texture = { name: 'case-B' };
    h.textures[1].callback(texture);
    assert.equal(h.run('visualizer.terrainMesh.material.map.name'), 'case-B');
});
