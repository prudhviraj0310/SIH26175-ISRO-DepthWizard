/**
 * ISRO SAC Geospatial 3D Workstation — Flythrough & Terrain Engine
 * ===============================================================
 * High-performance WebGL surface visualization engine designed for
 * ISRO Space Applications Centre (SAC) cartographic standards.
 * 
 * Features:
 * - Analytical Hillshade (NW 315°, Alt 45°) & Hypsometric Relief Draping
 * - True-Scale Geomorphic Slope Angle & Contour Maps
 * - Interactive 2D Elevation Profile Cross-Section Transect
 * - Real-Time Topographic Cursor Spatial Probe (Elev, Slope, Aspect)
 * - Calibrated Vertical Exaggeration (1.0x Natural Scale to 3.0x)
 * - Diurnal Sun Position Simulation (Azimuth & Altitude)
 * - Disaster Operations: Flood Surface Simulation & Helipad Landing Zones
 */

class FlythroughEngine {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.terrainMesh = null;
        this.waterMesh = null;
        
        // Lighting System
        this.sunLight = null;
        this.hemiLight = null;
        this.fillLight = null;
        this.sunAzimuthDeg = 315; // NW standard cartographic illumination
        this.sunAltitudeDeg = 45; // 45 deg solar elevation

        // Camera Orbit System (Z is UP)
        this.target = new THREE.Vector3(0, 0, 15);
        this.orbitRadius = 220;
        this.orbitYaw = -Math.PI / 2; // South looking North
        this.orbitPitch = 0.85;       // ~49 deg oblique aerial perspective
        this.mode = 'orbit';

        // Mouse & Input State
        this.mouseState = { isDown: false, button: 0, prevX: 0, prevY: 0 };
        this.raycaster = new THREE.Raycaster();
        this.mouseVec = new THREE.Vector2();

        // Shading Materials Dictionary
        this.materials = {};
        this.currentShading = 'relief'; // DEFAULT: Hypsometric Relief (Hillshaded)
        this.currentExaggeration = 1.0;
        this.planeDim = 180.0;
        this.groundDimM = 256.0; // 512 x 0.5m GSD

        // Caliper & Transect Measurement
        this.caliperActive = false;
        this.caliperPoints = [];
        this.caliperVisuals = [];
        this.landingZoneMarkers = [];

        // Cached active payload
        this.currentPayload = null;

        // Flythrough & Foundation State
        this.isFlying = false;
        this.flythroughTime = 0;
        this.flightAltitude = 36.0;
        this.skirtMesh = null;
        this.plinthMesh = null;

        this.init();
    }

    init() {
        const width = this.container.clientWidth || window.innerWidth;
        const height = this.container.clientHeight || (window.innerHeight - 56);

        // 1. Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf1f5f9);
        this.scene.fog = new THREE.FogExp2(0xf1f5f9, 0.0008);

        // 2. Camera (Z is UP)
        this.camera = new THREE.PerspectiveCamera(48, width / height, 1.0, 3500);
        this.camera.up.set(0, 0, 1);
        this.updateCameraOrbit();

        // 3. WebGL Renderer with High-Precision Shadows & ACES Tone Mapping
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true,
            powerPreference: 'high-performance'
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.1;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // 4. Cartographic Lighting System
        // Soft sky-ground hemisphere bounce
        this.hemiLight = new THREE.HemisphereLight(0xffffff, 0xcbd5e1, 0.82);
        this.scene.add(this.hemiLight);

        // Key Directional Sun Light (Positioned at 315° NW, 45° Alt)
        this.sunLight = new THREE.DirectionalLight(0xfffbeb, 1.35);
        this.updateSunPosition();
        this.sunLight.castShadow = true;
        this.sunLight.shadow.mapSize.width = 2048;
        this.sunLight.shadow.mapSize.height = 2048;
        this.sunLight.shadow.camera.near = 10;
        this.sunLight.shadow.camera.far = 700;
        this.sunLight.shadow.camera.left = -140;
        this.sunLight.shadow.camera.right = 140;
        this.sunLight.shadow.camera.top = 140;
        this.sunLight.shadow.camera.bottom = -140;
        this.sunLight.shadow.bias = -0.0005;
        this.scene.add(this.sunLight);

        // Subtle ambient fill light
        this.fillLight = new THREE.DirectionalLight(0x64748b, 0.3);
        this.fillLight.position.set(150, -150, 100);
        this.scene.add(this.fillLight);

        // 5. Dynamic Water Plane for Flood Inundation Analysis
        const waterGeo = new THREE.PlaneGeometry(this.planeDim * 1.25, this.planeDim * 1.25);
        const waterMat = new THREE.MeshStandardMaterial({
            color: 0x0284c7,
            roughness: 0.12,
            metalness: 0.85,
            transparent: true,
            opacity: 0.72,
            side: THREE.DoubleSide
        });
        this.waterMesh = new THREE.Mesh(waterGeo, waterMat);
        this.waterMesh.position.z = -100; // Hidden initially
        this.scene.add(this.waterMesh);

        // 6. Bind User Interactions
        this.bindEvents();

        // 7. Start Render Loop
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    updateSunPosition() {
        if (!this.sunLight) return;
        const azRad = (this.sunAzimuthDeg * Math.PI) / 180.0;
        const altRad = (this.sunAltitudeDeg * Math.PI) / 180.0;
        const dist = 260.0;

        // In Z-up coordinate system:
        // Azimuth 0 = North (+Y), 90 = East (+X), 180 = South (-Y), 270 = West (-X), 315 = NW (-X, +Y)
        const x = dist * Math.cos(altRad) * Math.sin(azRad);
        const y = dist * Math.cos(altRad) * Math.cos(azRad);
        const z = dist * Math.sin(altRad);

        this.sunLight.position.set(x, y, z);
    }

    updateCameraOrbit() {
        const cosPitch = Math.cos(this.orbitPitch);
        const sinPitch = Math.sin(this.orbitPitch);
        const cosYaw = Math.cos(this.orbitYaw);
        const sinYaw = Math.sin(this.orbitYaw);

        const x = this.target.x + this.orbitRadius * cosPitch * cosYaw;
        const y = this.target.y + this.orbitRadius * cosPitch * sinYaw;
        const z = this.target.z + this.orbitRadius * sinPitch;

        this.camera.position.set(x, y, z);
        this.camera.lookAt(this.target);
    }

    bindEvents() {
        // Keyboard Drone & Flythrough Controls
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isFlying) {
                this.toggleFlythrough();
                return;
            }
            const step = 4.0;
            if (e.code === 'KeyW' || e.code === 'ArrowUp') {
                const fwd = new THREE.Vector3();
                this.camera.getWorldDirection(fwd);
                this.camera.position.addScaledVector(fwd, step);
                this.target.addScaledVector(fwd, step);
            } else if (e.code === 'KeyS' || e.code === 'ArrowDown') {
                const fwd = new THREE.Vector3();
                this.camera.getWorldDirection(fwd);
                this.camera.position.addScaledVector(fwd, -step);
                this.target.addScaledVector(fwd, -step);
            } else if (e.code === 'KeyA' || e.code === 'ArrowLeft') {
                const right = new THREE.Vector3();
                const up = new THREE.Vector3(0, 0, 1);
                this.camera.getWorldDirection(right);
                right.cross(up).normalize();
                this.camera.position.addScaledVector(right, -step);
                this.target.addScaledVector(right, -step);
            } else if (e.code === 'KeyD' || e.code === 'ArrowRight') {
                const right = new THREE.Vector3();
                const up = new THREE.Vector3(0, 0, 1);
                this.camera.getWorldDirection(right);
                right.cross(up).normalize();
                this.camera.position.addScaledVector(right, step);
                this.target.addScaledVector(right, step);
            } else if (e.code === 'KeyQ') {
                this.camera.position.z = Math.max(8, this.camera.position.z - step);
            } else if (e.code === 'KeyE') {
                this.camera.position.z += step;
            }
        });

        window.addEventListener('resize', () => {
            const w = this.container.clientWidth;
            const h = this.container.clientHeight;
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(w, h);
        });

        // Prevent context menu on right click
        this.renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());

        // Mouse Down
        this.renderer.domElement.addEventListener('mousedown', (e) => {
            this.mouseState.isDown = true;
            this.mouseState.button = e.button;
            this.mouseState.prevX = e.clientX;
            this.mouseState.prevY = e.clientY;

            if (this.caliperActive && e.button === 0) {
                this.handleCaliperClick(e);
            }
        });

        window.addEventListener('mouseup', () => {
            this.mouseState.isDown = false;
        });

        // Mouse Move (Orbit, Pan & Spatial Probe)
        this.renderer.domElement.addEventListener('mousemove', (e) => {
            // 1. If dragging, update camera
            if (this.mouseState.isDown) {
                const dx = e.clientX - this.mouseState.prevX;
                const dy = e.clientY - this.mouseState.prevY;
                this.mouseState.prevX = e.clientX;
                this.mouseState.prevY = e.clientY;

                if (this.mouseState.button === 0) {
                    // Left-drag: Orbit Rotation
                    this.orbitYaw -= dx * 0.0055;
                    // Clamp pitch between 14° (oblique) and 87° (near nadir) - Never underground!
                    this.orbitPitch = Math.max(0.24, Math.min(1.52, this.orbitPitch + dy * 0.0055));
                    this.updateCameraOrbit();
                } else if (this.mouseState.button === 2) {
                    // Right-drag: Pan target
                    const panSpeed = this.orbitRadius * 0.0011;
                    const right = new THREE.Vector3();
                    const up = new THREE.Vector3(0, 0, 1);
                    this.camera.getWorldDirection(right);
                    right.cross(up).normalize();

                    const forward = new THREE.Vector3().copy(right).cross(up).normalize();
                    this.target.addScaledVector(right, -dx * panSpeed);
                    this.target.addScaledVector(forward, dy * panSpeed);
                    this.updateCameraOrbit();
                }
            }

            // 2. Real-Time Spatial Probe on hover
            this.handleSpatialProbe(e);
        });

        // Mouse Wheel Zoom
        this.renderer.domElement.addEventListener('wheel', (e) => {
            const zoomFactor = e.deltaY > 0 ? 1.08 : 0.92;
            this.orbitRadius = Math.max(35, Math.min(500, this.orbitRadius * zoomFactor));
            this.updateCameraOrbit();
        }, { passive: true });
    }

    handleSpatialProbe(e) {
        if (!this.terrainMesh || !this.currentPayload) return;
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouseVec.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouseVec.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouseVec, this.camera);
        const hits = this.raycaster.intersectObject(this.terrainMesh);

        const probeElem = document.getElementById('probe-telemetry');
        if (!probeElem) return;

        if (hits.length > 0) {
            const pt = hits[0].point;
            const gridS = this.currentPayload.grid_size;
            
            // Map 3D point back to grid indices
            const u = Math.max(0, Math.min(1, (pt.x + this.planeDim / 2) / this.planeDim));
            const v = Math.max(0, Math.min(1, 1.0 - (pt.y + this.planeDim / 2) / this.planeDim));
            const col = Math.min(gridS - 1, Math.floor(u * gridS));
            const row = Math.min(gridS - 1, Math.floor(v * gridS));
            const idx = row * gridS + col;

            const elev = this.currentPayload.metric_heights ? this.currentPayload.metric_heights[idx] : 0;
            const slope = this.currentPayload.slope_degrees ? this.currentPayload.slope_degrees[idx] : 0;
            
            let slopeClass = 'Flat';
            let slopeClassColor = '#10b981';
            if (slope >= 30) {
                slopeClass = 'Steep Escarpment';
                slopeClassColor = '#ef4444';
            } else if (slope >= 15) {
                slopeClass = 'Moderate';
                slopeClassColor = '#f59e0b';
            } else if (slope >= 5) {
                slopeClass = 'Gentle';
                slopeClassColor = '#eab308';
            }

            const groundX = (u * this.groundDimM).toFixed(1);
            const groundY = ((1.0 - v) * this.groundDimM).toFixed(1);

            probeElem.innerHTML = `
                <span>X: <b>${groundX}m</b></span>
                <span>Y: <b>${groundY}m</b></span>
                <span>Elev: <b style="color:#38bdf8;">${elev.toFixed(1)}m ASL</b></span>
                <span>Slope: <b style="color:${slopeClassColor};">${slope.toFixed(1)}° (${slopeClass})</b></span>
            `;
            probeElem.style.display = 'flex';
        } else {
            probeElem.style.display = 'none';
        }
    }

    loadTerrain(payload) {
        this.currentPayload = payload;
        const gridS = payload.grid_size;
        const normalizedZ = payload.normalized_z;
        const minZ = payload.min_elevation_m;
        const maxZ = payload.max_elevation_m;
        const elevRange = payload.elevation_range_m;

        // Clean up old terrain mesh
        if (this.terrainMesh) {
            this.scene.remove(this.terrainMesh);
            this.terrainMesh.geometry.dispose();
            if (this.terrainMesh.material) {
                if (Array.isArray(this.terrainMesh.material)) {
                    this.terrainMesh.material.forEach(m => m.dispose());
                } else {
                    this.terrainMesh.material.dispose();
                }
            }
            this.terrainMesh = null;
        }

        this.clearCaliper();
        this.clearLandingZones();

        // 1. Create Base Geometry
        const geometry = new THREE.PlaneGeometry(this.planeDim, this.planeDim, gridS - 1, gridS - 1);
        const positionAttr = geometry.attributes.position;

        // 2. Physical Metric Height Scaling
        // Scale factor: planeDim / groundDim (e.g. 180 / 256 = 0.703 Three.js units per ground meter)
        const metricScale = (this.planeDim / this.groundDimM);
        const baseZHeight = elevRange * metricScale;
        this.currentExaggeration = 1.0;

        let sumZ = 0;
        for (let i = 0; i < positionAttr.count; i++) {
            const zNorm = normalizedZ[i] || 0.0;
            // 1.0x Natural Scale: elevation difference translated directly to metric units
            const zVal = zNorm * baseZHeight * this.currentExaggeration;
            positionAttr.array[i * 3 + 2] = zVal;
            sumZ += zVal;
        }
        positionAttr.needsUpdate = true;
        geometry.computeVertexNormals();

        const meanElevationZ = sumZ / positionAttr.count;
        this.target.set(0, 0, meanElevationZ);
        this.buildPlinthAndSkirt(baseZHeight);

        // 3. Load all Cartographic Textures (Relief, Hillshade, Slope, Ortho)
        const loader = new THREE.TextureLoader();

        const setupTex = (url, onLoad) => {
            if (!url) return;
            loader.load(url, (tex) => {
                tex.wrapS = THREE.ClampToEdgeWrapping;
                tex.wrapT = THREE.ClampToEdgeWrapping;
                tex.generateMipmaps = true;
                onLoad(tex);
            });
        };

        // Initialize Material Collection
        this.materials = {
            relief: null,
            hillshade: null,
            slope: null,
            ortho_shaded: null,
            optical: null,
            wire: new THREE.MeshBasicMaterial({
                color: 0x38bdf8,
                wireframe: true
            })
        };

        // Create Mesh immediately with placeholder
        const placeholderMat = new THREE.MeshStandardMaterial({
            color: 0x1e293b,
            roughness: 0.8
        });
        this.terrainMesh = new THREE.Mesh(geometry, placeholderMat);
        this.terrainMesh.castShadow = true;
        this.terrainMesh.receiveShadow = true;
        this.scene.add(this.terrainMesh);

        // Load 1: Blended Hypsometric Relief (Default)
        setupTex(payload.relief_texture_url, (tex) => {
            this.materials.relief = new THREE.MeshStandardMaterial({
                map: tex,
                roughness: 0.70,
                metalness: 0.04,
                side: THREE.DoubleSide
            });
            if (this.currentShading === 'relief') {
                this.terrainMesh.material = this.materials.relief;
            }
        });

        // Load 2: Analytical Hillshade
        setupTex(payload.hillshade_texture_url, (tex) => {
            this.materials.hillshade = new THREE.MeshStandardMaterial({
                map: tex,
                roughness: 0.85,
                metalness: 0.02,
                side: THREE.DoubleSide
            });
            if (this.currentShading === 'hillshade') {
                this.terrainMesh.material = this.materials.hillshade;
            }
        });

        // Load 3: Slope Classification Map
        setupTex(payload.slope_texture_url, (tex) => {
            this.materials.slope = new THREE.MeshStandardMaterial({
                map: tex,
                roughness: 0.75,
                metalness: 0.04,
                side: THREE.DoubleSide
            });
            if (this.currentShading === 'slope') {
                this.terrainMesh.material = this.materials.slope;
            }
        });

        // Load 4: Ortho + Hillshade Hybrid
        setupTex(payload.ortho_hillshade_url, (tex) => {
            this.materials.ortho_shaded = new THREE.MeshStandardMaterial({
                map: tex,
                roughness: 0.80,
                metalness: 0.05,
                side: THREE.DoubleSide
            });
            if (this.currentShading === 'ortho_shaded') {
                this.terrainMesh.material = this.materials.ortho_shaded;
            }
        });

        // Load 5: Raw Optical Satellite Image
        setupTex(payload.texture_data_url, (tex) => {
            this.materials.optical = new THREE.MeshStandardMaterial({
                map: tex,
                roughness: 0.85,
                metalness: 0.05,
                side: THREE.DoubleSide
            });
            if (this.currentShading === 'optical') {
                this.terrainMesh.material = this.materials.optical;
            }
        });

        // Load 6: Live Error Difference Map (|DSM_AI - DSM_LiDAR|)
        if (payload.error_texture_url) {
            setupTex(payload.error_texture_url, (tex) => {
                this.materials.error = new THREE.MeshStandardMaterial({
                    map: tex,
                    roughness: 0.75,
                    metalness: 0.04,
                    side: THREE.DoubleSide
                });
                if (this.currentShading === 'error') {
                    this.terrainMesh.material = this.materials.error;
                }
            });
        }

        // 4. Update Legends & HUD
        this.updateElevationLegend(minZ, maxZ);

        // 5. Reset camera to natural oblique view
        this.orbitRadius = 220;
        this.orbitYaw = -Math.PI / 2;
        this.orbitPitch = 0.85;
        this.updateCameraOrbit();

        // 6. Reset water mesh
        if (this.waterMesh) {
            this.waterMesh.position.z = -100;
        }
    }

    buildPlinthAndSkirt(baseZHeight) {
        if (this.skirtMesh) {
            this.scene.remove(this.skirtMesh);
            if (this.skirtMesh.geometry) this.skirtMesh.geometry.dispose();
            this.skirtMesh = null;
        }
        if (this.plinthMesh) {
            this.scene.remove(this.plinthMesh);
            if (this.plinthMesh.geometry) this.plinthMesh.geometry.dispose();
            this.plinthMesh = null;
        }

        if (!this.currentPayload) return;

        const gridSize = this.currentPayload.grid_size || 128;
        const normalizedZ = this.currentPayload.normalized_z;
        const baseZ = -6.0; // Plinth base floor
        const dim = this.planeDim; // 180.0
        const step = dim / (gridSize - 1);
        const half = dim / 2;

        const vertices = [];
        const uvs = [];

        const addQuad = (x1, y1, z1, x2, y2, z2) => {
            vertices.push(
                x1, y1, z1,   x1, y1, baseZ,  x2, y2, baseZ,
                x1, y1, z1,   x2, y2, baseZ,  x2, y2, z2
            );
            uvs.push(
                0, 1,  0, 0,  1, 0,
                0, 1,  1, 0,  1, 1
            );
        };

        const getZ = (r, c) => {
            const idx = r * gridSize + c;
            return (normalizedZ[idx] || 0.0) * baseZHeight * this.currentExaggeration;
        };

        // 1. Top Edge (y = half)
        for (let c = 0; c < gridSize - 1; c++) {
            const x1 = -half + c * step;
            const x2 = -half + (c + 1) * step;
            addQuad(x1, half, getZ(0, c), x2, half, getZ(0, c + 1));
        }

        // 2. Right Edge (x = half)
        for (let r = 0; r < gridSize - 1; r++) {
            const y1 = half - r * step;
            const y2 = half - (r + 1) * step;
            addQuad(half, y1, getZ(r, gridSize - 1), half, y2, getZ(r + 1, gridSize - 1));
        }

        // 3. Bottom Edge (y = -half)
        for (let c = gridSize - 1; c > 0; c--) {
            const x1 = -half + c * step;
            const x2 = -half + (c - 1) * step;
            addQuad(x1, -half, getZ(gridSize - 1, c), x2, -half, getZ(gridSize - 1, c - 1));
        }

        // 4. Left Edge (x = -half)
        for (let r = gridSize - 1; r > 0; r--) {
            const y1 = half - r * step;
            const y2 = half - (r - 1) * step;
            addQuad(-half, y1, getZ(r, 0), -half, y2, getZ(r - 1, 0));
        }

        const skirtGeo = new THREE.BufferGeometry();
        skirtGeo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        skirtGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
        skirtGeo.computeVertexNormals();

        const skirtMat = new THREE.MeshStandardMaterial({
            color: 0xe2e8f0,
            roughness: 0.88,
            metalness: 0.04,
            side: THREE.DoubleSide
        });
        this.skirtMesh = new THREE.Mesh(skirtGeo, skirtMat);
        this.skirtMesh.receiveShadow = true;
        this.scene.add(this.skirtMesh);

        // Solid museum-grade beveled architectural plinth
        const plinthGeo = new THREE.BoxGeometry(dim + 8.0, dim + 8.0, 3.5);
        const plinthMat = new THREE.MeshStandardMaterial({
            color: 0xcbd5e1,
            roughness: 0.85,
            metalness: 0.08
        });
        this.plinthMesh = new THREE.Mesh(plinthGeo, plinthMat);
        this.plinthMesh.position.set(0, 0, baseZ - 1.75);
        this.plinthMesh.receiveShadow = true;
        this.scene.add(this.plinthMesh);
    }

    updateElevationLegend(minZ, maxZ) {
        const legendMin = document.getElementById('legend-min-elev');
        const legendMax = document.getElementById('legend-max-elev');
        const legendMid = document.getElementById('legend-mid-elev');
        if (legendMin) legendMin.innerText = `${minZ.toFixed(1)}m`;
        if (legendMax) legendMax.innerText = `${maxZ.toFixed(1)}m`;
        if (legendMid) legendMid.innerText = `${((minZ + maxZ) / 2).toFixed(1)}m`;
    }

    setShadingMode(mode) {
        this.currentShading = mode;
        if (!this.terrainMesh) return;

        // Toggle legend visibility based on mode
        const elevLegend = document.getElementById('elev-legend-container');
        const slopeLegend = document.getElementById('slope-legend-container');
        const errorLegend = document.getElementById('error-legend-container');
        if (elevLegend) elevLegend.style.display = (mode === 'relief' || mode === 'optical' || mode === 'ortho_shaded') ? 'flex' : 'none';
        if (slopeLegend) slopeLegend.style.display = (mode === 'slope') ? 'flex' : 'none';
        if (errorLegend) errorLegend.style.display = (mode === 'error') ? 'flex' : 'none';

        if (mode === 'relief' && this.materials.relief) {
            this.terrainMesh.material = this.materials.relief;
        } else if (mode === 'hillshade' && this.materials.hillshade) {
            this.terrainMesh.material = this.materials.hillshade;
        } else if (mode === 'slope' && this.materials.slope) {
            this.terrainMesh.material = this.materials.slope;
        } else if (mode === 'ortho_shaded' && this.materials.ortho_shaded) {
            this.terrainMesh.material = this.materials.ortho_shaded;
        } else if (mode === 'optical' && this.materials.optical) {
            this.terrainMesh.material = this.materials.optical;
        } else if (mode === 'error' && this.materials.error) {
            this.terrainMesh.material = this.materials.error;
        } else if (mode === 'wire' && this.materials.wire) {
            this.terrainMesh.material = this.materials.wire;
        }
    }

    setExaggeration(val) {
        if (!this.terrainMesh || !this.currentPayload) return;
        this.currentExaggeration = val;
        const normalizedZ = this.currentPayload.normalized_z;
        const elevRange = this.currentPayload.elevation_range_m;
        const metricScale = (this.planeDim / this.groundDimM);
        const baseZHeight = elevRange * metricScale;

        const posAttr = this.terrainMesh.geometry.attributes.position;
        let sumZ = 0;
        for (let i = 0; i < posAttr.count; i++) {
            const zNorm = normalizedZ[i] || 0.0;
            const zVal = zNorm * baseZHeight * this.currentExaggeration;
            posAttr.array[i * 3 + 2] = zVal;
            sumZ += zVal;
        }
        posAttr.needsUpdate = true;
        this.terrainMesh.geometry.computeVertexNormals();
        this.buildPlinthAndSkirt(baseZHeight);

        const meanElevationZ = sumZ / posAttr.count;
        this.target.z = meanElevationZ;
        this.updateCameraOrbit();
    }

    setCameraPreset(preset) {
        if (preset === 'nadir') {
            this.orbitPitch = 1.50; // ~86° top-down nadir
            this.orbitRadius = 210;
        } else if (preset === 'oblique') {
            this.orbitPitch = 0.85; // ~49° oblique
            this.orbitYaw = -Math.PI / 2;
            this.orbitRadius = 220;
        } else if (preset === 'profile') {
            this.orbitPitch = 0.28; // ~16° surface level grazing profile
            this.orbitRadius = 230;
        } else if (preset === 'reset') {
            this.target.set(0, 0, this.target.z);
            this.orbitPitch = 0.85;
            this.orbitYaw = -Math.PI / 2;
            this.orbitRadius = 220;
        }
        this.updateCameraOrbit();
    }

    setCaliperTool(active) {
        this.caliperActive = active;
        const caliperStatus = document.getElementById('caliper-status');
        const crosshair = document.getElementById('crosshair');
        if (active) {
            if (caliperStatus) caliperStatus.innerText = 'CLICK POINT A ON TERRAIN';
            if (caliperStatus) caliperStatus.style.color = '#38bdf8';
            if (crosshair) crosshair.style.display = 'block';
            this.container.style.cursor = 'crosshair';
        } else {
            if (caliperStatus) caliperStatus.innerText = 'STANDBY';
            if (caliperStatus) caliperStatus.style.color = 'var(--text-muted)';
            if (crosshair) crosshair.style.display = 'none';
            this.container.style.cursor = 'default';
        }
    }

    handleCaliperClick(e) {
        if (!this.terrainMesh || !this.currentPayload) return;
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouseVec.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouseVec.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouseVec, this.camera);
        const hits = this.raycaster.intersectObject(this.terrainMesh);

        if (hits.length > 0) {
            const hit = hits[0];
            const pt = hit.point.clone();

            // Place 3D beacon sphere
            const sphereGeo = new THREE.SphereGeometry(1.4, 20, 20);
            const sphereMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
            const sphere = new THREE.Mesh(sphereGeo, sphereMat);
            sphere.position.copy(pt);
            this.scene.add(sphere);
            this.caliperVisuals.push(sphere);

            this.caliperPoints.push(pt);

            const caliperStatus = document.getElementById('caliper-status');
            if (this.caliperPoints.length === 1) {
                if (caliperStatus) caliperStatus.innerText = 'POINT A SET. CLICK POINT B';
            } else if (this.caliperPoints.length === 2) {
                const p1 = this.caliperPoints[0];
                const p2 = this.caliperPoints[1];

                // Draw connecting laser line
                const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
                const lineMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, linewidth: 3 });
                const line = new THREE.Line(lineGeo, lineMat);
                this.scene.add(line);
                this.caliperVisuals.push(line);

                // Calculate accurate real ground metrics
                const elevRange = this.currentPayload.elevation_range_m;
                const minElev = this.currentPayload.min_elevation_m;
                const metricScale = (this.planeDim / this.groundDimM);
                const baseZHeight = elevRange * metricScale;

                const normZ1 = p1.z / (baseZHeight * this.currentExaggeration + 1e-6);
                const normZ2 = p2.z / (baseZHeight * this.currentExaggeration + 1e-6);
                const h1 = minElev + normZ1 * elevRange;
                const h2 = minElev + normZ2 * elevRange;
                const deltaH = Math.abs(h2 - h1);

                // Planimetric 2D ground distance in meters
                const dX = (p2.x - p1.x) / metricScale;
                const dY = (p2.y - p1.y) / metricScale;
                const groundDistM = Math.sqrt(dX * dX + dY * dY);

                // 3D Euclidean distance
                const dist3D = Math.sqrt(groundDistM * groundDistM + deltaH * deltaH);

                // Slope in degrees and gradient percentage
                const slopeDeg = Math.atan2(deltaH, Math.max(groundDistM, 0.1)) * (180.0 / Math.PI);
                const slopePct = (deltaH / Math.max(groundDistM, 0.1)) * 100.0;

                // Update UI Readouts
                const elH = document.getElementById('reading-height');
                const elDist = document.getElementById('reading-distance');
                const elDist3D = document.getElementById('reading-distance-3d');
                const elSlope = document.getElementById('reading-slope');
                const elGrade = document.getElementById('reading-grade');

                if (elH) elH.innerText = `${deltaH.toFixed(2)} m`;
                if (elDist) elDist.innerText = `${groundDistM.toFixed(1)} m`;
                if (elDist3D) elDist3D.innerText = `${dist3D.toFixed(1)} m`;
                if (elSlope) elSlope.innerText = `${slopeDeg.toFixed(1)}°`;
                if (elGrade) elGrade.innerText = `${slopePct.toFixed(1)}%`;

                // Draw authentic 2D profile cross section sampled from DSM
                this.drawElevationProfile(p1, p2, h1, h2, groundDistM);

                if (caliperStatus) caliperStatus.innerText = 'TRANSECT COMPUTED';
                this.caliperPoints = [];
                this.setCaliperTool(false);
            }
        }
    }

    drawElevationProfile(p1, p2, h1, h2, groundDistM) {
        const canvas = document.getElementById('profile-canvas');
        if (!canvas || !this.currentPayload) return;
        canvas.style.display = 'block';
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);

        // Clean white dashboard background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, w, h);

        // Sample 30 real points from metric_heights array
        const gridS = this.currentPayload.grid_size;
        const heights = this.currentPayload.metric_heights || [];
        const numSamples = 32;
        const sampledH = [];

        const u1 = (p1.x + this.planeDim / 2) / this.planeDim;
        const v1 = 1.0 - (p1.y + this.planeDim / 2) / this.planeDim;
        const u2 = (p2.x + this.planeDim / 2) / this.planeDim;
        const v2 = 1.0 - (p2.y + this.planeDim / 2) / this.planeDim;

        for (let i = 0; i <= numSamples; i++) {
            const t = i / numSamples;
            const u = u1 + (u2 - u1) * t;
            const v = v1 + (v2 - v1) * t;
            const col = Math.min(gridS - 1, Math.max(0, Math.floor(u * gridS)));
            const row = Math.min(gridS - 1, Math.max(0, Math.floor(v * gridS)));
            const elev = heights[row * gridS + col] || (h1 + (h2 - h1) * t);
            sampledH.push(elev);
        }

        const minProfileH = Math.min(...sampledH);
        const maxProfileH = Math.max(...sampledH);
        const rangeH = Math.max(1.0, maxProfileH - minProfileH);

        // Draw grid lines
        ctx.strokeStyle = '#f1f5f9';
        ctx.lineWidth = 1;
        for (let y = 15; y < h - 15; y += 15) {
            ctx.beginPath();
            ctx.moveTo(35, y);
            ctx.lineTo(w - 15, y);
            ctx.stroke();
        }

        // Draw profile fill
        const padX = 40;
        const padY = 18;
        const graphW = w - padX - 15;
        const graphH = h - padY * 2;

        ctx.beginPath();
        ctx.moveTo(padX, h - padY);
        for (let i = 0; i <= numSamples; i++) {
            const x = padX + (i / numSamples) * graphW;
            const normVal = (sampledH[i] - minProfileH) / rangeH;
            const y = (h - padY) - normVal * graphH;
            ctx.lineTo(x, y);
        }
        ctx.lineTo(padX + graphW, h - padY);
        ctx.closePath();

        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, 'rgba(37, 99, 235, 0.25)');
        grad.addColorStop(1, 'rgba(37, 99, 235, 0.01)');
        ctx.fillStyle = grad;
        ctx.fill();

        // Draw profile stroke line
        ctx.beginPath();
        for (let i = 0; i <= numSamples; i++) {
            const x = padX + (i / numSamples) * graphW;
            const normVal = (sampledH[i] - minProfileH) / rangeH;
            const y = (h - padY) - normVal * graphH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Text Labels
        ctx.fillStyle = '#64748b';
        ctx.font = '9px SF Mono, monospace';
        ctx.fillText(`${maxProfileH.toFixed(0)}m`, 4, padY + 6);
        ctx.fillText(`${minProfileH.toFixed(0)}m`, 4, h - padY);
        ctx.fillText(`0m`, padX, h - 4);
        ctx.fillText(`${groundDistM.toFixed(0)}m`, w - 35, h - 4);
    }

    clearCaliper() {
        if (this.caliperVisuals) {
            this.caliperVisuals.forEach(v => this.scene.remove(v));
            this.caliperVisuals = [];
        }
        this.caliperPoints = [];
        const canvas = document.getElementById('profile-canvas');
        if (canvas) canvas.style.display = 'none';
    }

    setWaterLevel(levelNormalized) {
        if (!this.waterMesh || !this.currentPayload) return;
        if (levelNormalized <= 0.001) {
            this.waterMesh.position.z = -100;
        } else {
            const elevRange = this.currentPayload.elevation_range_m;
            const metricScale = (this.planeDim / this.groundDimM);
            const baseZHeight = elevRange * metricScale;
            this.waterMesh.position.z = levelNormalized * baseZHeight * this.currentExaggeration;
        }
    }

    setSunAzimuth(azimuthDeg) {
        this.sunAzimuthDeg = azimuthDeg;
        this.updateSunPosition();
    }

    setSunAltitude(altDeg) {
        this.sunAltitudeDeg = altDeg;
        this.updateSunPosition();
    }

    renderLandingZones(zones) {
        this.clearLandingZones();
        if (!zones || zones.length === 0 || !this.currentPayload) return;

        const elevRange = this.currentPayload.elevation_range_m;
        const minZ = this.currentPayload.min_elevation_m;
        const metricScale = (this.planeDim / this.groundDimM);
        const baseZHeight = elevRange * metricScale;

        zones.forEach(z => {
            const group = new THREE.Group();
            
            // Glowing landing circle
            const ringGeo = new THREE.RingGeometry(3.0, 4.2, 32);
            const ringMat = new THREE.MeshBasicMaterial({ color: 0x10b981, side: THREE.DoubleSide });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            group.add(ring);

            // Vertical beacon
            const cylGeo = new THREE.CylinderGeometry(0.35, 0.35, 8, 16);
            const cylMat = new THREE.MeshBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.85 });
            const beacon = new THREE.Mesh(cylGeo, cylMat);
            beacon.rotation.x = Math.PI / 2;
            beacon.position.z = 4;
            group.add(beacon);

            const worldX = (z.norm_x - 0.5) * this.planeDim;
            const worldY = (z.norm_y - 0.5) * this.planeDim;
            const normZ = (z.elevation_m - minZ) / (elevRange + 1e-6);
            const worldZ = normZ * baseZHeight * this.currentExaggeration + 0.6;

            group.position.set(worldX, worldY, worldZ);
            this.scene.add(group);
            this.landingZoneMarkers.push(group);
        });
    }

    clearLandingZones() {
        if (this.landingZoneMarkers) {
            this.landingZoneMarkers.forEach(m => this.scene.remove(m));
            this.landingZoneMarkers = [];
        }
    }

    takeSnapshot() {
        this.renderer.render(this.scene, this.camera);
        const dataUrl = this.renderer.domElement.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = `ISRO_SAC_3D_Surface_${Date.now()}.png`;
        link.href = dataUrl;
        link.click();
    }

    toggleFlythrough() {
        this.isFlying = !this.isFlying;
        const btn = document.getElementById('btn-cam-fly');
        const banner = document.getElementById('flythrough-banner');
        if (this.isFlying) {
            this.flythroughTime = 0;
            if (btn) {
                btn.classList.add('active');
                btn.innerHTML = '⏹ Stop Flight';
            }
            if (banner) banner.style.display = 'block';
        } else {
            if (btn) {
                btn.classList.remove('active');
                btn.innerHTML = '🎥 Aerial Flythrough';
            }
            if (banner) banner.style.display = 'none';
            this.updateCameraOrbit();
        }
        return this.isFlying;
    }

    animate() {
        requestAnimationFrame(this.animate);
        if (this.isFlying && this.currentPayload) {
            this.flythroughTime += 0.0035;
            const radX = 85.0;
            const radY = 78.0;
            const camX = radX * Math.cos(this.flythroughTime);
            const camY = radY * Math.sin(this.flythroughTime * 1.35) * 0.75;
            const meanZ = this.target.z || 15.0;
            const camZ = meanZ + this.flightAltitude + 7.0 * Math.sin(this.flythroughTime * 2.0);

            this.camera.position.set(camX, camY, camZ);
            const lookAhead = this.flythroughTime + 0.16;
            const lookX = radX * Math.cos(lookAhead) * 0.35;
            const lookY = radY * Math.sin(lookAhead * 1.35) * 0.35;
            const lookZ = meanZ + 3.0;
            this.camera.lookAt(lookX, lookY, lookZ);
        }
        this.renderer.render(this.scene, this.camera);
    }
}
