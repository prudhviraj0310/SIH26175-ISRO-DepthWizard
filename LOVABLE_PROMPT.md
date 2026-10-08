# LOVABLE MASTER PROMPT: DepthWizard — Monocular Satellite Height Reconstruction Workstation

Copy and paste everything below into **Lovable** (or Claude / Cursor) to generate the complete, production-grade frontend application.

---

```markdown
You are building **DepthWizard**, an elite, high-performance desktop GIS & cartographic workstation for **Single-Image Satellite Height Reconstruction (ISRO SAC Problem Statement 26175)**.

### CORE OBJECTIVE
DepthWizard reconstructs metric 3D Digital Surface Models (DSM) and terrain elevation from a single monocular optical satellite image (Maxar VHR 0.5m, Sentinel-2 10m, or Synthetic). It calibrates relative depth predictions (Depth Anything v2) using sparse reference terrain anchors (SRTM / Copernicus GLO-30 / FABDEM) into accurate real-world metric heights.

Build a complete, standalone, production-grade React + Vite + Tailwind CSS + Lucide Icons + Three.js web application that looks and functions like a world-class scientific workstation (similar to QGIS, Blender CAD, and Palantir Foundry).

---

### DESIGN SYSTEM & AESTHETICS (STRICT GIS WORKSTATION STANDARD)
1. **Color Palette**:
   - Background Base: `#110f0e` (warm deep obsidian)
   - Surface / Cards: `#181614` with subtle borders `#282522` and hover `#35312c`
   - Foreground Text: `#e8e5df` (primary), `#9e9b94` (muted), `#5a5752` (subtle)
   - Accent & Status:
     - Metric Tier 2 (Calibrated): `#5e9872` (sage emerald)
     - Metric Tier 1 (DEM Baseline): `#d97706` (warm amber)
     - Interactive / Active: `#38bdf8` (crisp cyan)
     - Critical / Alert: `#ef4444` (ruby red)
2. **Typography**:
   - Headings & Interface: `Geist`, `Inter`, or system sans-serif (`font-sans`)
   - Data readouts, coordinates, telemetry, HUD, and code: `Geist Mono` or `JetBrains Mono` (`font-mono`)
3. **No AI Slop / No Cheesy Gradients**:
   - Do NOT use purple/pink glowing gradients or generic templates.
   - Use micro-borders (1px solid `#282522`), subtle backdrop blurs (`backdrop-blur-md`), precise 8px/12px border radiuses, and high-density, professional layouts.

---

### APPLICATION ARCHITECTURE & PAGES

#### 1. GLOBAL COLLAPSIBLE SIDEBAR (shadcn/ui Sidebar-07 Style)
- Left navigation bar with logo mark ("D" badge) and brand title "Depth Wizard".
- Pin / Lock toggle icon (Locked = permanently docked; Unlocked = collapses to icon rail).
- Two top tabs:
  - **PAGES**:
    - `DW Studio` (Main Creation & Processing Workbench - Route: `/workbench`)
    - `3D Explorer` (Interactive WebGL 3D Terrain Inspector - Route: `/explorer`)
    - `Research & Docs` (ISRO SAC Spec, Math, Benchmarks, Architecture - Route: `/docs`)
  - **JOBS**:
    - Active job indicator ("No jobs running" or "Processing scene...").
    - Job count badge, "+ Generate New" button, and list of Recent & Saved reconstruction runs with timestamps, runtime, and MAE/RMSE metrics.

---

#### 2. PAGE 1: DW STUDIO (`/workbench`)
This is the primary scene ingest, configuration, and processing workstation.

##### A. HUD Header
- Coordinate Readout: `24.7863° N, 85.9412° E` (updates dynamically based on selected scene).
- Full-width subtle canvas latitude/longitude ruler tick marks.
- Cartographic Footnote: `Datum: WGS84 · EPSG:4326 · Elevation Reference: EGM96`.

##### B. Input Configuration View (Pre-Execution)
- **Left Column (Source Selector)**:
  - Mode Tabs: `Curated Library`, `Search Online (Sentinel/CDSE)`, `Custom Upload (GeoTIFF/PNG)`.
  - Filter Pills:
    - Collection: `All`, `Maxar VHR (0.5m)`, `Sentinel-2 (10m)`, `Synthetic Benchmark`.
    - Terrain Type: `Any terrain`, `Hilly / Mountain`, `Urban / Built-up`, `Coastal / Delta`.
  - Interactive Scene Cards:
    - Card displays high-resolution thumbnail, scene title (e.g. "DC_04_23 Urban District", "ISRO SAC Campus Ahmedabad", "Highland Ridge"), GSD resolution badge (`0.5m`), CRS code (`EPSG:32618`), and terrain category.
    - Clicking a card instantly selects it as active.
- **Right Column (Live Scene Inspector)**:
  - High-resolution optical preview viewport with an interactive **Hover Magnifier Loupe** (1.75x circular zoom lens following the cursor).
  - Spatial telemetry tags:
    - Bounding Box & Center Coordinates
    - Ground Footprint (e.g., `1.28 km × 1.28 km`)
    - Native Resolution (`0.5 m/px`)
    - Anchor Reference (`Copernicus GLO-30 / SRTM`)
  - Calibration Tier Badge:
    - `TIER 2: METRIC RECONSTRUCTION` (Fine-tuned Depth Anything v2 + Huber Loss Anchor, Metric surface with terrain anchor).
  - Execution Action: Large button `▶ START RECONSTRUCTION`.
    - Clicking switches to animated progress state ("Loading model checkpoint...", "Predicting relative depth...", "Solving metric affine transformation...", "Generating 3D surface mesh...").

##### C. Processing Results Grid (Post-Execution)
Once reconstruction finishes (or when viewing a completed job), show a split workbench:
- **Left 60%**:
  - Selected Scene Card with metadata summary.
  - Live calculation log terminal with realistic pipeline execution logs.
  - Four seamless tabbed output displays:
    1. **Source Optical RGB**: Clean original sensor imagery.
    2. **Relative Depth (DAV2)**: Monocular relative depth colormap (spectral / inferno).
    3. **Surface Elevation (DSM)**: Absolute metric heightmap with elevation legend (min/max meters).
    4. **3D Reconstructed Mesh**: Wireframe/shaded preview.
- **Right 40%**:
  - Live interactive 3D miniature preview.
  - Action button: `LAUNCH 3D EXPLORER WORKSTATION →` (switches to Page 2).

---

#### 3. PAGE 2: 3D EXPLORER (`/explorer`)
A full-screen, high-performance 3D cartographic viewer powered by Three.js.

##### A. 3D WebGL Canvas Features
- High-precision terrain mesh with ground plinth (solid skirt below terrain).
- Lighting: Directional Sun Light (default 315° NW, 45° altitude) with soft shadows and cartographic hemisphere skylight.
- Camera Controls: Orbit (left click), Pan (right click), Zoom (wheel), and Top-Down Ortho reset button.
- Shading Material Selector:
  - `Hypsometric Relief`: Cartographic elevation color ramp with analytical hillshading.
  - `True-Color Optical Drape`: Satellite RGB image mapped over the 3D surface.
  - `Geomorphic Slope Angle`: Steepness map (green = flat, yellow = moderate, red = cliff > 35°).
  - `Wireframe Overlay`: Semi-transparent structural polygon wireframe.

##### B. Floating Toolbars & Controls
1. **Terrain Dynamics Drawer**:
   - Vertical Exaggeration Slider (`1.0x` to `3.0x`).
   - Sun Azimuth & Solar Altitude Sliders (simulate diurnal daylight progression).
2. **Flood Risk Simulation**:
   - Interactive Water Elevation Slider (meters above datum).
   - Real-time animated transparent water plane intersecting terrain.
   - Live telemetry readout: Submerged Area (`km²`) and `% Inundated Surface`.
3. **Interactive 2-Point Caliper & Transect Tool**:
   - "Measure Caliper" toggle: clicking two points on the 3D terrain drops interactive markers and renders a connecting 3D line.
   - Computes:
     - 3D Spatial Distance (`m`)
     - Ground Horizontal Distance (`m`)
     - Height Differential `|ΔH|` (`m`)
     - Mean Gradient (`%`)
   - Opens a floating bottom drawer with a **2D Elevation Cross-Section Transect Graph** plotted across the cut line.
4. **Topographic Spatial Probe**:
   - Hovering the mouse over the terrain displays real-time pointer coordinates, Elevation (`Z = ...m`), and Slope (`...°`).

---

#### 4. PAGE 3: RESEARCH & DOCS (`/docs`)
Comprehensive scientific and technical documentation.

1. **Problem Statement Briefing**:
   - ISRO SAC Problem Statement 26175: Monocular Satellite Height Reconstruction.
   - Comparison of Traditional Stereo Photogrammetry (requires dual-angle passes, fails on moving objects/clouds) vs. Monocular Deep Depth Estimation.
2. **Mathematical Pipeline Formulation**:
   - Stage 1: Zero-shot relative depth estimation $d(u,v) = \mathcal{M}_{\text{DAV2}}(I)$.
   - Stage 2: Robust Huber-loss affine calibration to sparse elevation anchors:
     $$\min_{a, b} \sum_{(u,v)} \rho_\delta \left( (a \cdot d(u,v) + b) - H_{\text{DEM}}(u,v) \right)$$
   - Stage 3: High-frequency residual detail transfer and Poisson surface reconstruction.
3. **Validation & Benchmark Matrix**:
   - Interactive data table comparing models on DFC2019 (Jacksonville / Omaha), ISRO SAC Ahmedabad, and Alpine Relief datasets.
   - Metrics: MAE (m), RMSE (m), AbsRel, and Accuracy threshold ($\delta < 1.25$).
4. **Export Center**:
   - Download Buttons:
     - `GeoTIFF 32-bit Float DSM`
     - `Wavefront OBJ Mesh + MTL`
     - `Validation Telemetry CSV`
     - `High-Res Ortho PNG`

---

### MOCK DATA & API LAYER
- Build a robust `api.js` client.
- Provide full, rich mock data for all curated scenes, elevations, meshes, and benchmark metrics so that the entire app works flawlessly out of the box in Lovable without needing any external server.
- Also include support for connecting to the local DepthWizard backend at `http://127.0.0.1:8099/api` (endpoints: `/reconstruct`, `/jobs`, `/export`) with automatic graceful fallback to mock data.

Build this complete, pixel-perfect application with all components, styles, and full interactivity now.
```
