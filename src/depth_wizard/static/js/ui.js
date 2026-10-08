/**
 * DepthWizard — Advanced Geospatial Workstation Controller (v8.0)
 * ==============================================================
 * Powers:
 * 1. Collapsible workspace sidebar (shadcn/ui layout) with drag rail and tabs
 * 2. Multi-collection and multi-terrain catalog filtering
 * 3. Geodetic coordinate HUD ruler & 3× optical hover magnifying inspection
 * 4. Modular .xp-box accordion panels with micro-transitions
 * 5. Automated flood inundation rise simulation loop
 * 6. Multi-job tabs bar for simultaneous 3D project sessions
 */

(function () {
  'use strict';

  // --- CATALOG OF SCENES & METADATA ---
  const SCENE_CATALOG = [
    {
      id: "gamus_dc_04_23",
      name: "DC_04_23 Urban District",
      collection: "vhr",
      terrain: "urban",
      gsd: "0.5m",
      crs: "EPSG:32618 (UTM 18N)",
      coords: "38.8951° N, 77.0364° W",
      source: "GAMUS LiDAR Benchmark / High-Res Optical",
      thumb: "/static/data/dc_sample_rgb.png",
      tier: "tier-2",
      tierName: "TIER 2: METRIC ESTIMATE",
      tierDesc: "Fine-tuned DAv2 on VHR imagery. Metric surface anchored to airborne LiDAR."
    },
    {
      id: "isro_sac_ahmedabad",
      name: "ISRO SAC Campus (Ahmedabad)",
      collection: "synthetic",
      terrain: "urban",
      gsd: "1.0m",
      crs: "EPSG:32643 (UTM 43N)",
      coords: "23.0225° N, 72.5714° E",
      source: "ISRO Procedural Benchmark Mesh",
      thumb: "/static/data/dc_sample_rgb.png",
      tier: "tier-2",
      tierName: "TIER 2: PROCEDURAL FIXTURE",
      tierDesc: "Synthetic geometric evaluation fixture for building height testing."
    },
    {
      id: "synthetic_hills_64",
      name: "Highland Ridge (Synthetic)",
      collection: "synthetic",
      terrain: "hilly",
      gsd: "2.0m",
      crs: "Local Relative Datum",
      coords: "31.1048° N, 77.1734° E",
      source: "Analytical Sine/Cosine Surface",
      thumb: "/static/data/dc_sample_rgb.png",
      tier: "tier-1",
      tierName: "TIER 1: RELATIVE SURFACE",
      tierDesc: "Controlled synthetic relief fixture for slope and escarpment testing."
    }
  ];

  let currentActiveSceneId = "gamus_dc_04_23";
  let activeCollectionFilter = "all";
  let activeTerrainFilter = "all";
  let openJobs = new Set(["gamus_dc_04_23"]);
  let floodSimInterval = null;

  // --- ROUTING: Workbench / Explorer / Docs ---
  const ROUTES = ["workbench", "explorer", "docs"];

  function getRoute() {
    const hash = window.location.hash.replace(/^#\/?/, '').toLowerCase();
    return ROUTES.includes(hash) ? hash : "workbench";
  }

  function setRoute(route, updateHash = true) {
    if (!ROUTES.includes(route)) route = "workbench";
    if (updateHash) {
      window.location.hash = `#/${route}`;
    }

    document.querySelectorAll('.page-view').forEach(p => {
      p.classList.toggle('active', p.id === `page-${route}`);
    });

    document.querySelectorAll('.sb-page-link').forEach(link => {
      link.classList.toggle('active', link.dataset.route === route);
    });

    const canvas = document.getElementById('flythrough-canvas-container');
    if (canvas) {
      if (route === 'explorer') {
        canvas.style.display = 'block';
        canvas.style.opacity = '1';
        canvas.style.pointerEvents = 'auto';
      } else {
        canvas.style.display = 'none';
        canvas.style.opacity = '0';
        canvas.style.pointerEvents = 'none';
      }
    }

    window.dispatchEvent(new Event('resize'));
  }

  window.addEventListener('hashchange', () => setRoute(getRoute(), false));

  // --- COLLAPSIBLE SIDEBAR & RESIZE RAIL ---
  const sidebar = document.getElementById('app-sidebar');
  const lockBtn = document.getElementById('sb-lock-toggle');
  const rail = document.getElementById('sb-rail');
  let sidebarLocked = true;

  lockBtn?.addEventListener('click', () => {
    sidebarLocked = !sidebarLocked;
    sidebar.classList.toggle('is-collapsed', !sidebarLocked);
    lockBtn.style.color = sidebarLocked ? 'var(--accent)' : 'var(--text-dim)';
  });

  // Sidebar Tabs: Pages vs Jobs
  const tabPages = document.getElementById('sb-tab-pages');
  const tabJobs = document.getElementById('sb-tab-jobs');
  const navPages = document.getElementById('sb-nav-pages');
  const jobsView = document.getElementById('sb-jobs-view');

  tabPages?.addEventListener('click', () => {
    tabPages.classList.add('active');
    tabJobs.classList.remove('active');
    navPages.style.display = 'flex';
    jobsView.style.display = 'none';
  });

  tabJobs?.addEventListener('click', () => {
    tabJobs.classList.add('active');
    tabPages.classList.remove('active');
    navPages.style.display = 'none';
    jobsView.style.display = 'flex';
  });

  // Sidebar drag rail resize
  let isResizing = false;
  rail?.addEventListener('mousedown', (e) => {
    isResizing = true;
    document.body.style.cursor = 'col-resize';
  });
  window.addEventListener('mousemove', (e) => {
    if (!isResizing) return;
    const newWidth = Math.max(180, Math.min(420, e.clientX));
    document.documentElement.style.setProperty('--sb-width', `${newWidth}px`);
  });
  window.addEventListener('mouseup', () => {
    if (isResizing) {
      isResizing = false;
      document.body.style.cursor = 'default';
      window.dispatchEvent(new Event('resize'));
    }
  });

  // Navigation Links
  document.querySelectorAll('.sb-page-link').forEach(link => {
    link.addEventListener('click', () => {
      setRoute(link.dataset.route);
    });
  });

  // --- WORKBENCH: SOURCE TABS & CATALOG FILTERING ---
  const sourceTabs = document.querySelectorAll('.source-tab-btn');
  sourceTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      sourceTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.dataset.source;
      document.querySelectorAll('.source-pane').forEach(p => {
        p.classList.toggle('active', p.id === `pane-${target}`);
      });
    });
  });

  function renderCatalog() {
    const container = document.getElementById('catalog-list');
    if (!container) return;
    container.innerHTML = '';

    const filtered = SCENE_CATALOG.filter(s => {
      const matchCol = activeCollectionFilter === 'all' || s.collection === activeCollectionFilter;
      const matchTer = activeTerrainFilter === 'all' || s.terrain === activeTerrainFilter;
      return matchCol && matchTer;
    });

    filtered.forEach(scene => {
      const card = document.createElement('div');
      card.className = `catalog-item ${scene.id === currentActiveSceneId ? 'active' : ''}`;
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span class="catalog-name">${scene.name}</span>
          <span class="badge ${scene.collection === 'vhr' ? 'badge-vhr' : scene.collection === 'synthetic' ? 'badge-synth' : 'badge-lidar'}">${scene.collection}</span>
        </div>
        <div class="catalog-meta">
          <span>${scene.terrain} · ${scene.gsd}</span>
          <span>${scene.crs.split(' ')[0]}</span>
        </div>
      `;
      card.addEventListener('click', () => selectCatalogScene(scene));
      container.appendChild(card);
    });
  }

  function selectCatalogScene(scene) {
    currentActiveSceneId = scene.id;
    renderCatalog();

    // Update HUD
    const hudCoords = document.getElementById('hud-lat-lon');
    if (hudCoords) hudCoords.innerText = scene.coords;

    // Update Stage Preview
    const stageImg = document.getElementById('stage-preview-img');
    if (stageImg) stageImg.src = scene.thumb;

    // Update Tier Banner
    const tierPill = document.getElementById('tier-indicator-pill');
    const tierDesc = document.getElementById('tier-description-text');
    if (tierPill) {
      tierPill.className = `tier-pill ${scene.tier}`;
      tierPill.innerText = scene.tierName;
    }
    if (tierDesc) tierDesc.innerText = scene.tierDesc;

    // Synchronize selectElem for app.js
    const selectElem = document.getElementById('scene-selector');
    if (selectElem) {
      selectElem.value = scene.id;
      selectElem.dispatchEvent(new Event('change'));
    }
  }

  // Collection Filter Chips
  document.querySelectorAll('[data-collection]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-collection]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeCollectionFilter = btn.dataset.collection;
      renderCatalog();
    });
  });

  // Terrain Filter Chips
  document.querySelectorAll('[data-terrain]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-terrain]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeTerrainFilter = btn.dataset.terrain;
      renderCatalog();
    });
  });

  // Draw Geodetic HUD Ruler Canvas
  function drawHudRuler() {
    const canvas = document.getElementById('hud-ruler');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.font = '8px monospace';

    ctx.beginPath();
    ctx.moveTo(0, h - 2);
    ctx.lineTo(w, h - 2);
    ctx.stroke();

    for (let x = 0; x < w; x += 12) {
      const isMajor = x % 48 === 0;
      ctx.beginPath();
      ctx.moveTo(x, h - 2);
      ctx.lineTo(x, h - (isMajor ? 10 : 5));
      ctx.stroke();
      if (isMajor && x > 0 && x < w - 20) {
        ctx.fillText(`${x / 10}km`, x - 8, h - 11);
      }
    }
  }

  // Interactive 3x Hover Magnifier
  function bindMagnifier(stageId, lensId, imgId) {
    const stage = document.getElementById(stageId);
    const lens = document.getElementById(lensId);
    const img = document.getElementById(imgId);
    if (!stage || !lens || !img) return;

    stage.addEventListener('mouseenter', () => {
      lens.style.display = 'block';
      lens.style.backgroundImage = `url(${img.src})`;
      lens.style.backgroundRepeat = 'no-repeat';
    });

    stage.addEventListener('mouseleave', () => {
      lens.style.display = 'none';
    });

    stage.addEventListener('mousemove', (e) => {
      const rect = stage.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const lensRadius = 60;
      lens.style.left = `${x - lensRadius}px`;
      lens.style.top = `${y - lensRadius}px`;

      const zoom = 3;
      const bgX = -((x * zoom) - lensRadius);
      const bgY = -((y * zoom) - lensRadius);
      lens.style.backgroundSize = `${rect.width * zoom}px ${rect.height * zoom}px`;
      lens.style.backgroundPosition = `${bgX}px ${bgY}px`;
    });
  }

  // --- START RECONSTRUCTION ACTION ---
  document.getElementById('btn-start-generation-main')?.addEventListener('click', () => {
    addJobTab(currentActiveSceneId);
    setRoute('explorer');
  });

  // --- 3D EXPLORER: MULTI-JOB TABS ---
  function addJobTab(sceneId) {
    openJobs.add(sceneId);
    const scene = SCENE_CATALOG.find(s => s.id === sceneId) || { name: sceneId };
    const container = document.getElementById('job-tabs-container');
    if (!container) return;

    container.innerHTML = '';
    openJobs.forEach(id => {
      const s = SCENE_CATALOG.find(item => item.id === id) || { name: id };
      const tab = document.createElement('button');
      tab.className = `job-tab ${id === currentActiveSceneId ? 'active' : ''}`;
      tab.type = 'button';
      tab.innerHTML = `<span class="status-dot"></span> ${s.name}`;
      tab.addEventListener('click', () => {
        currentActiveSceneId = id;
        document.querySelectorAll('.job-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const selectElem = document.getElementById('scene-selector');
        if (selectElem) {
          selectElem.value = id;
          selectElem.dispatchEvent(new Event('change'));
        }
      });
      container.appendChild(tab);
    });

    const jobCount = document.getElementById('jobs-count-text');
    if (jobCount) jobCount.innerText = openJobs.size;
  }

  document.getElementById('btn-new-job-tab')?.addEventListener('click', () => {
    setRoute('workbench');
  });

  // --- MODULAR .XP-BOX ACCORDION CONTROLS ---
  document.querySelectorAll('.xp-head').forEach(head => {
    head.addEventListener('click', () => {
      const box = head.closest('.xp-box');
      if (box) box.classList.toggle('is-open');
    });
  });

  // --- DISASTER & SCENARIO ANALYSIS CONSOLE ---
  const tabFlood = document.getElementById('tab-hazard-flood');
  const tabLandslide = document.getElementById('tab-hazard-landslide');
  const tabHlz = document.getElementById('tab-hazard-hlz');

  const panelFlood = document.getElementById('subpanel-flood');
  const panelLandslide = document.getElementById('subpanel-landslide');
  const panelHlz = document.getElementById('subpanel-hlz');

  function setHazardTab(tab) {
    [tabFlood, tabLandslide, tabHlz].forEach(t => t?.classList.remove('active'));
    [panelFlood, panelLandslide, panelHlz].forEach(p => { if (p) p.style.display = 'none'; });

    if (tab === 'flood') {
      tabFlood?.classList.add('active');
      if (panelFlood) panelFlood.style.display = 'flex';
    } else if (tab === 'landslide') {
      tabLandslide?.classList.add('active');
      if (panelLandslide) panelLandslide.style.display = 'flex';
    } else if (tab === 'hlz') {
      tabHlz?.classList.add('active');
      if (panelHlz) panelHlz.style.display = 'flex';
    }
  }

  tabFlood?.addEventListener('click', () => setHazardTab('flood'));
  tabLandslide?.addEventListener('click', () => setHazardTab('landslide'));
  tabHlz?.addEventListener('click', () => setHazardTab('hlz'));

  // Automated Flood Surge Simulation Loop
  const btnFloodPlay = document.getElementById('btn-flood-sim-play');
  const btnFloodReset = document.getElementById('btn-flood-sim-reset');
  const floodSlider = document.getElementById('slider-flood-surge');

  btnFloodPlay?.addEventListener('click', () => {
    if (floodSimInterval) {
      clearInterval(floodSimInterval);
      floodSimInterval = null;
      btnFloodPlay.innerText = '▶ Simulate Surge';
      btnFloodPlay.classList.add('primary');
    } else {
      btnFloodPlay.innerText = '⏸ Pause Simulation';
      btnFloodPlay.classList.remove('primary');
      floodSimInterval = setInterval(() => {
        if (!floodSlider) return;
        let val = parseFloat(floodSlider.value) + 0.5;
        if (val > 15.0) val = 0.0;
        floodSlider.value = val;
        floodSlider.dispatchEvent(new Event('input'));
        floodSlider.dispatchEvent(new Event('change'));
      }, 500);
    }
  });

  btnFloodReset?.addEventListener('click', () => {
    if (floodSimInterval) {
      clearInterval(floodSimInterval);
      floodSimInterval = null;
      if (btnFloodPlay) {
        btnFloodPlay.innerText = '▶ Simulate Surge';
        btnFloodPlay.classList.add('primary');
      }
    }
    if (floodSlider) {
      floodSlider.value = 0;
      floodSlider.dispatchEvent(new Event('input'));
      floodSlider.dispatchEvent(new Event('change'));
    }
  });

  // --- SYNC LIVE TERRAIN PROBE WITH TELEMETRY ---
  const probeContainer = document.getElementById('probe-telemetry');
  const probeElevReadout = document.getElementById('probe-elev-readout');
  const probeSlopeReadout = document.getElementById('probe-slope-readout');

  if (probeContainer && (probeElevReadout || probeSlopeReadout)) {
    const probeObserver = new MutationObserver(() => {
      const text = probeContainer.innerText;
      const elevMatch = text.match(/Surface:\s*([^\n\r]+)/);
      const slopeMatch = text.match(/Slope:\s*([^\n\r]+)/);
      if (elevMatch && probeElevReadout) probeElevReadout.innerText = elevMatch[1];
      if (slopeMatch && probeSlopeReadout) probeSlopeReadout.innerText = slopeMatch[1];
    });
    probeObserver.observe(probeContainer, { childList: true, subtree: true, characterData: true });
  }

  // --- DRAWER CALIPER TOGGLE ---
  const btnToggleCaliper = document.getElementById('btn-toggle-caliper');
  const drawerCaliper = document.getElementById('drawer-caliper');
  btnToggleCaliper?.addEventListener('click', () => {
    const isHidden = drawerCaliper.hidden;
    drawerCaliper.hidden = !isHidden;
    btnToggleCaliper.classList.toggle('active', !isHidden);
  });
  document.querySelector('[data-close="drawer-caliper"]')?.addEventListener('click', () => {
    if (drawerCaliper) drawerCaliper.hidden = true;
    btnToggleCaliper?.classList.remove('active');
  });

  // --- THEME SWITCHER ---
  const themeToggle = document.getElementById('global-theme-toggle');
  themeToggle?.addEventListener('click', () => {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
  });

  // --- PRESET CHIP BUTTONS IN SEARCH PANE ---
  document.querySelectorAll('.live-preset-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const place = chip.dataset.place;
      const input = document.getElementById('live-place-input');
      const btnBuild = document.getElementById('btn-live-auto-process');
      if (input && btnBuild) {
        input.value = place;
        btnBuild.click();
      }
    });
  });

  // --- INITIALIZATION ---
  window.addEventListener('DOMContentLoaded', () => {
    renderCatalog();
    drawHudRuler();
    bindMagnifier('inspect-stage', 'stage-lens', 'stage-preview-img');
    bindMagnifier('mini-inspect-stage', 'mini-lens', 'mini-inspect-img');
    addJobTab('gamus_dc_04_23');
    setRoute(getRoute(), false);
  });

})();
