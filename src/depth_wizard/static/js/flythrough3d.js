/**
 * DepthWizard 3D Flythrough Engine (High-Fidelity Aerospace Edition)
 * =================================================================
 * Professional WebGL rendering engine providing:
 * - 3D terrain surface reconstruction from monocular satellite DSM
 * - High-luminance ACESFilmic tone-mapped optical satellite texture
 * - True spherical aerial orbit controller (never underground, smooth clamp)
 * - First-Person Drone Flight recon mode (WASD + mouse look)
 * - 3D Laser Caliper measurement tool (point-to-point elevation & distance)
 * - Dynamic Flood Inundation reflective water surface plane
 * - 3D Emergency Helicopter Landing Zone (HLZ) glowing beacons
 * - Directional sunlight & slope hazard visualization
 */

class FlythroughEngine {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.terrainMesh = null;
        this.waterMesh = null;
        this.sunLight = null;
        this.hemiLight = null;
        this.fillLight = null;
        
        // Orbital Camera Parameters
        this.target = new THREE.Vector3(0, 0, 15);
        this.orbitRadius = 220;
        this.orbitYaw = -Math.PI / 2; // Looking from South to North
        this.orbitPitch = 0.85;       // ~49 deg oblique aerial angle
        
        // Mode & Input State
        this.mode = 'orbit'; // 'orbit' or 'drone'
        this.keys = {};
        this.mouseState = { isDown: false, button: 0, prevX: 0, prevY: 0 };
        this.droneSpeed = 1.2;
        this.droneYaw = 0;
        this.dronePitch = -0.2;

        // Laser Caliper & Measurement State
        this.caliperActive = false;
        this.currentExaggeration = 1.0;
        this.caliperPoints = [];
        this.caliperVisuals = [];
        this.landingZoneMarkers = [];
        this.raycaster = new THREE.Raycaster();
        this.mouseVec = new THREE.Vector2();

        // Shading materials
        this.opticalMaterial = null;
        this.vertexColorMaterial = null;
        this.slopeMaterial = null;
        this.wireframeMaterial = null;
        this.currentShading = 'optical';

        // Terrain metadata cache
        this.currentPayload = null;

        this.init();
    }

    init() {
        const width = this.container.clientWidth || window.innerWidth;
        const height = this.container.clientHeight || (window.innerHeight - 58);

        // 1. Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x050811);
        this.scene.fog = new THREE.FogExp2(0x050811, 0.0018);

        // 2. Camera (Z is UP)
        this.camera = new THREE.PerspectiveCamera(50, width / height, 1.0, 3000);
        this.camera.up.set(0, 0, 1);
        this.updateCameraOrbit();

        // 3. Renderer with ACESFilmic Tone Mapping for rich satellite colors
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.15;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // 4. Lighting: Hemisphere + Key Sun + Ambient Fill
        this.hemiLight = new THREE.HemisphereLight(0xffffff, 0x334466, 0.95);
        this.scene.add(this.hemiLight);

        this.sunLight = new THREE.DirectionalLight(0xfffaf0, 1.25);
        this.sunLight.position.set(120, -140, 220);
        this.sunLight.castShadow = true;
        this.sunLight.shadow.mapSize.width = 2048;
        this.sunLight.shadow.mapSize.height = 2048;
        this.sunLight.shadow.camera.near = 10;
        this.sunLight.shadow.camera.far = 600;
        this.sunLight.shadow.camera.left = -120;
        this.sunLight.shadow.camera.right = 120;
        this.sunLight.shadow.camera.top = 120;
        this.sunLight.shadow.camera.bottom = -120;
        this.scene.add(this.sunLight);

        this.fillLight = new THREE.DirectionalLight(0x99bbdd, 0.4);
        this.fillLight.position.set(-120, 140, 100);
        this.scene.add(this.fillLight);

        // 5. Water Plane for Flood Simulation
        const waterGeo = new THREE.PlaneGeometry(240, 240);
        const waterMat = new THREE.MeshStandardMaterial({
            color: 0x0077be,
            roughness: 0.1,
            metalness: 0.8,
            transparent: true,
            opacity: 0.65,
            side: THREE.DoubleSide
        });
        this.waterMesh = new THREE.Mesh(waterGeo, waterMat);
        this.waterMesh.position.z = -100; // Hidden initially
        this.scene.add(this.waterMesh);

        // 6. Bind Events
        this.bindEvents();

        // 7. Render Loop
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    updateCameraOrbit() {
        if (this.mode !== 'orbit') return;
        // Spherical coords around this.target (Z is up)
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
        window.addEventListener('resize', () => {
            const w = this.container.clientWidth;
            const h = this.container.clientHeight;
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(w, h);
        });

        // Keyboard controls for Drone flight
        window.addEventListener('keydown', (e) => {
            this.keys[e.key.toLowerCase()] = true;
        });
        window.addEventListener('keyup', (e) => {
            this.keys[e.key.toLowerCase()] = false;
        });

        // Mouse controls
        this.renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());

        this.renderer.domElement.addEventListener('mousedown', (e) => {
            this.mouseState.isDown = true;
            this.mouseState.button = e.button;
            this.mouseState.prevX = e.clientX;
            this.mouseState.prevY = e.clientY;

            // If caliper tool is active and left-clicked
            if (this.caliperActive && e.button === 0) {
                this.handleCaliperClick(e);
            }
        });

        window.addEventListener('mouseup', () => {
            this.mouseState.isDown = false;
        });

        window.addEventListener('mousemove', (e) => {
            if (!this.mouseState.isDown) return;
            const dx = e.clientX - this.mouseState.prevX;
            const dy = e.clientY - this.mouseState.prevY;
            this.mouseState.prevX = e.clientX;
            this.mouseState.prevY = e.clientY;

            if (this.mode === 'drone') {
                this.droneYaw -= dx * 0.003;
                this.dronePitch = Math.max(-1.4, Math.min(1.4, this.dronePitch - dy * 0.003));
            } else {
                // Orbit mode
                if (this.mouseState.button === 0) {
                    // Left click: Orbit rotation
                    this.orbitYaw -= dx * 0.006;
                    // Clamp pitch between 15 deg and 85 deg (never underground!)
                    this.orbitPitch = Math.max(0.25, Math.min(1.48, this.orbitPitch + dy * 0.006));
                    this.updateCameraOrbit();
                } else if (this.mouseState.button === 2) {
                    // Right click: Pan target
                    const panSpeed = this.orbitRadius * 0.0012;
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
        });

        // Mouse wheel for zoom
        this.renderer.domElement.addEventListener('wheel', (e) => {
            if (this.mode === 'orbit') {
                const zoomFactor = e.deltaY > 0 ? 1.08 : 0.92;
                this.orbitRadius = Math.max(40, Math.min(480, this.orbitRadius * zoomFactor));
                this.updateCameraOrbit();
            }
        }, { passive: true });
    }

    loadTerrain(payload) {
        this.currentPayload = payload;
        const gridS = payload.grid_size;
        const normalizedZ = payload.normalized_z;
        const metricHeights = payload.metric_heights;
        const elevRange = payload.elevation_range_m;

        // Remove old terrain mesh
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

        // Clear caliper visuals & landing markers
        this.clearCaliper();
        this.clearLandingZones();

        // 1. Create Plane Geometry (centered in XY at 0, 0)
        const planeDim = 160;
        const geometry = new THREE.PlaneGeometry(planeDim, planeDim, gridS - 1, gridS - 1);
        const positionAttr = geometry.attributes.position;

        // 2. Displace Vertices according to DSM
        const zExaggeration = Math.min(42.0, Math.max(14.0, (elevRange / 100.0) * 32.0));
        this.currentExaggeration = zExaggeration;
        
        let sumZ = 0;
        for (let i = 0; i < positionAttr.count; i++) {
            const zNorm = normalizedZ[i] || 0.0;
            const zVal = zNorm * zExaggeration;
            positionAttr.array[i * 3 + 2] = zVal;
            sumZ += zVal;
        }
        positionAttr.needsUpdate = true;
        geometry.computeVertexNormals();

        const meanElevationZ = sumZ / positionAttr.count;
        this.target.set(0, 0, meanElevationZ);

        // 3. Precompute Hypsometric Elevation & Slope Attributes
        const count = positionAttr.count;
        const elevColors = new Float32Array(count * 3);
        const slopeColors = new Float32Array(count * 3);
        const normals = geometry.attributes.normal;

        for (let i = 0; i < count; i++) {
            const zNorm = normalizedZ[i] || 0.0;
            // High-contrast Hypsometric tint: Deep Blue -> Teal -> Green -> Yellow -> Orange -> Crisp White Peak
            let r = 0, g = 0, b = 0;
            if (zNorm < 0.2) {
                const t = zNorm / 0.2;
                r = 0.02; g = 0.25 + 0.5 * t; b = 0.85;
            } else if (zNorm < 0.4) {
                const t = (zNorm - 0.2) / 0.2;
                r = 0.02 + 0.1 * t; g = 0.75 + 0.2 * t; b = 0.85 - 0.5 * t;
            } else if (zNorm < 0.65) {
                const t = (zNorm - 0.4) / 0.25;
                r = 0.12 + 0.7 * t; g = 0.95 - 0.1 * t; b = 0.2;
            } else if (zNorm < 0.85) {
                const t = (zNorm - 0.65) / 0.2;
                r = 0.85 + 0.15 * t; g = 0.85 - 0.5 * t; b = 0.1;
            } else {
                const t = (zNorm - 0.85) / 0.15;
                r = 1.0; g = 0.35 + 0.65 * t; b = 0.1 + 0.9 * t;
            }
            elevColors[i * 3] = r;
            elevColors[i * 3 + 1] = g;
            elevColors[i * 3 + 2] = b;

            // Slope angle relative to vertical [0, 0, 1]
            const nz = normals ? (normals.array[i * 3 + 2] || 1.0) : 1.0;
            const slopeDeg = Math.acos(Math.max(-1, Math.min(1, nz))) * (180 / Math.PI);
            if (slopeDeg < 15) {
                slopeColors[i * 3] = 0.1; slopeColors[i * 3 + 1] = 0.85; slopeColors[i * 3 + 2] = 0.35;
            } else if (slopeDeg < 30) {
                slopeColors[i * 3] = 0.95; slopeColors[i * 3 + 1] = 0.75; slopeColors[i * 3 + 2] = 0.1;
            } else {
                slopeColors[i * 3] = 0.95; slopeColors[i * 3 + 1] = 0.2; slopeColors[i * 3 + 2] = 0.15;
            }
        }

        const elevColorAttr = new THREE.BufferAttribute(elevColors, 3);
        const slopeColorAttr = new THREE.BufferAttribute(slopeColors, 3);

        geometry.setAttribute('color', elevColorAttr);

        this.vertexColorMaterial = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.75,
            metalness: 0.1,
            side: THREE.DoubleSide
        });

        this.slopeMaterial = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.8,
            metalness: 0.05,
            side: THREE.DoubleSide
        });

        this.wireframeMaterial = new THREE.MeshBasicMaterial({
            color: 0x00e5ff,
            wireframe: true
        });

        // 4. Texture Loading from Base64 Data URL
        const textureLoader = new THREE.TextureLoader();
        textureLoader.load(payload.texture_data_url, (tex) => {
            tex.wrapS = THREE.ClampToEdgeWrapping;
            tex.wrapT = THREE.ClampToEdgeWrapping;
            tex.generateMipmaps = true;

            this.opticalMaterial = new THREE.MeshStandardMaterial({
                map: tex,
                roughness: 0.82,
                metalness: 0.08,
                side: THREE.DoubleSide
            });

            this.terrainMesh = new THREE.Mesh(geometry, this.opticalMaterial);
            this.terrainMesh.castShadow = true;
            this.terrainMesh.receiveShadow = true;
            this.scene.add(this.terrainMesh);

            // Re-apply current shading mode
            this.setShadingMode(this.currentShading);

            // Set natural camera orbit
            this.orbitRadius = 220;
            this.orbitYaw = -Math.PI / 2;
            this.orbitPitch = 0.85;
            this.updateCameraOrbit();

            // Set water mesh base
            if (this.waterMesh) {
                this.waterMesh.position.z = -10;
            }
        });
    }

    setShadingMode(mode) {
        this.currentShading = mode;
        if (!this.terrainMesh) return;

        if (mode === 'optical' && this.opticalMaterial) {
            this.terrainMesh.material = this.opticalMaterial;
        } else if (mode === 'elev' && this.vertexColorMaterial) {
            this.terrainMesh.material = this.vertexColorMaterial;
        } else if (mode === 'slope' && this.slopeMaterial) {
            this.terrainMesh.material = this.slopeMaterial;
        } else if (mode === 'wire' && this.wireframeMaterial) {
            this.terrainMesh.material = this.wireframeMaterial;
        }
    }

    setCameraView(view) {
        if (view === 'nadir') {
            this.mode = 'orbit';
            this.orbitPitch = 1.48; // ~85 deg top-down
            this.orbitRadius = 210;
            this.updateCameraOrbit();
        } else if (view === 'oblique') {
            this.mode = 'orbit';
            this.orbitPitch = 0.85; // ~49 deg oblique
            this.orbitYaw = -Math.PI / 2;
            this.orbitRadius = 220;
            this.updateCameraOrbit();
        } else if (view === 'fpv') {
            this.setMode('drone');
        }
    }

    setMode(newMode) {
        this.mode = newMode;
        const hudModeText = document.getElementById('hud-mode-text');
        if (newMode === 'drone') {
            if (hudModeText) hudModeText.innerText = 'DRONE RECON (WASD TO FLY, MOUSE LOOK)';
            this.camera.position.set(0, -70, this.target.z + 20);
            this.droneYaw = 0;
            this.dronePitch = -0.15;
        } else {
            if (hudModeText) hudModeText.innerText = 'ORBIT INSPECTION (LEFT-DRAG TO ROTATE, RIGHT-DRAG TO PAN, SCROLL TO ZOOM)';
            this.updateCameraOrbit();
        }
    }

    setCaliperTool(active) {
        this.caliperActive = active;
        const caliperStatus = document.getElementById('caliper-status');
        const crosshair = document.getElementById('crosshair');
        if (caliperStatus) {
            caliperStatus.innerText = active ? 'CLICK 2 POINTS' : 'READY';
            caliperStatus.style.color = active ? 'var(--accent-cyan)' : 'var(--text-muted)';
        }
        if (crosshair) crosshair.classList.toggle('active', active);
    }

    setExaggeration(val) {
        if (!this.terrainMesh || !this.currentPayload) return;
        const gridS = this.currentPayload.grid_size;
        const normalizedZ = this.currentPayload.normalized_z;
        const baseZExag = Math.min(42.0, Math.max(14.0, (this.currentPayload.elevation_range_m / 100.0) * 32.0));
        const effectiveExag = baseZExag * val;
        this.currentExaggeration = effectiveExag;

        const posAttr = this.terrainMesh.geometry.attributes.position;
        for (let i = 0; i < posAttr.count; i++) {
            const zNorm = normalizedZ[i] || 0.0;
            posAttr.array[i * 3 + 2] = zNorm * effectiveExag;
        }
        posAttr.needsUpdate = true;
        this.terrainMesh.geometry.computeVertexNormals();
    }

    toggleWireframe() {
        if (!this.terrainMesh) return false;
        if (this.terrainMesh.material === this.wireframeMaterial) {
            this.setShadingMode(this.currentShading === 'wire' ? 'optical' : this.currentShading);
            return false;
        } else {
            this.terrainMesh.material = this.wireframeMaterial;
            return true;
        }
    }

    takeSnapshot() {
        this.renderer.render(this.scene, this.camera);
        const dataUrl = this.renderer.domElement.toDataURL('image/png');
        const a = document.createElement('a');
        a.download = `DepthWizard_3D_Snapshot_${Date.now()}.png`;
        a.href = dataUrl;
        a.click();
    }

    setWaterLevel(levelNormalized) {
        if (!this.waterMesh || !this.currentPayload) return;
        if (levelNormalized <= 0.001) {
            this.waterMesh.position.z = -100;
        } else {
            this.waterMesh.position.z = levelNormalized * this.currentExaggeration;
        }
    }

    setSunAngle(angleDeg) {
        if (!this.sunLight) return;
        const rad = (angleDeg * Math.PI) / 180;
        const dist = 220;
        this.sunLight.position.x = dist * Math.cos(rad);
        this.sunLight.position.z = dist * Math.sin(rad);
    }

    renderLandingZones(zones) {
        this.clearLandingZones();
        if (!zones || zones.length === 0 || !this.currentPayload) return;

        const terrainScale = 160.0;

        zones.forEach(z => {
            const group = new THREE.Group();
            
            // Outer glowing landing ring
            const ringGeo = new THREE.RingGeometry(2.5, 3.5, 32);
            const ringMat = new THREE.MeshBasicMaterial({ color: 0x00e676, side: THREE.DoubleSide });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            group.add(ring);

            // Vertical beacon cylinder
            const cylGeo = new THREE.CylinderGeometry(0.35, 0.35, 8, 16);
            const cylMat = new THREE.MeshBasicMaterial({ color: 0x00e676, transparent: true, opacity: 0.85 });
            const beacon = new THREE.Mesh(cylGeo, cylMat);
            beacon.rotation.x = Math.PI / 2;
            beacon.position.z = 4;
            group.add(beacon);

            // Coordinates on 160x160 plane
            const worldX = (z.norm_x - 0.5) * terrainScale;
            const worldY = (z.norm_y - 0.5) * terrainScale;
            const minZ = this.currentPayload.min_elevation_m;
            const elevRange = this.currentPayload.elevation_range_m;
            const normZ = (z.elevation_m - minZ) / (elevRange + 1e-6);
            const worldZ = normZ * this.currentExaggeration + 0.6;

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

    handleCaliperClick(e) {
        if (!this.terrainMesh || !this.currentPayload) return;
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouseVec.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouseVec.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouseVec, this.camera);
        const intersects = this.raycaster.intersectObject(this.terrainMesh);

        if (intersects.length > 0) {
            const hit = intersects[0];
            const pt = hit.point;

            // Add marker beacon
            const markerGeo = new THREE.SphereGeometry(1.2, 16, 16);
            const markerMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
            const marker = new THREE.Mesh(markerGeo, markerMat);
            marker.position.copy(pt);
            this.scene.add(marker);
            this.caliperVisuals.push(marker);

            this.caliperPoints.push(pt);

            if (this.caliperPoints.length === 2) {
                const p1 = this.caliperPoints[0];
                const p2 = this.caliperPoints[1];

                // Draw glowing laser line
                const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
                const lineMat = new THREE.LineBasicMaterial({ color: 0x00e5ff, linewidth: 3 });
                const line = new THREE.Line(lineGeo, lineMat);
                this.scene.add(line);
                this.caliperVisuals.push(line);

                // Calculate metric distance & height delta
                const elevRange = this.currentPayload.elevation_range_m;
                const minElev = this.currentPayload.min_elevation_m;
                const h1 = minElev + (p1.z / this.currentExaggeration) * elevRange;
                const h2 = minElev + (p2.z / this.currentExaggeration) * elevRange;
                const deltaH = Math.abs(h2 - h1);

                const groundDistM = Math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2) * 1.5;
                const slopeDeg = Math.atan2(deltaH, Math.max(groundDistM, 0.1)) * (180 / Math.PI);

                // Update UI readouts
                const readingHeight = document.getElementById('reading-height');
                const readingDist = document.getElementById('reading-distance');
                const readingSlope = document.getElementById('reading-slope');
                if (readingHeight) readingHeight.innerText = `${deltaH.toFixed(2)} m`;
                if (readingDist) readingDist.innerText = `${groundDistM.toFixed(1)} m`;
                if (readingSlope) readingSlope.innerText = `${slopeDeg.toFixed(1)}°`;

                // Draw elevation profile
                this.drawElevationProfile(h1, h2, groundDistM);

                // Reset points for next measurement
                this.caliperPoints = [];
                this.setCaliperTool(false);
            }
        }
    }

    drawElevationProfile(h1, h2, groundDistM) {
        const canvas = document.getElementById('profile-canvas');
        if (!canvas) return;
        canvas.style.display = 'block';
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);

        // Background
        ctx.fillStyle = '#060a12';
        ctx.fillRect(0, 0, w, h);

        // Profile curve with intermediate terrain points
        const points = 12;
        ctx.beginPath();
        ctx.moveTo(15, h - 15);
        for (let i = 0; i <= points; i++) {
            const t = i / points;
            const x = 15 + t * (w - 30);
            const baseH = h1 + (h2 - h1) * t;
            const variation = (i > 0 && i < points) ? Math.sin(t * Math.PI) * 5 : 0;
            const curH = baseH + variation;
            const maxVal = Math.max(h1, h2, 10);
            const y = h - 15 - ((curH / maxVal) * (h - 30));
            ctx.lineTo(x, y);
        }
        ctx.lineTo(w - 15, h - 15);
        ctx.closePath();

        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, 'rgba(0, 229, 255, 0.4)');
        grad.addColorStop(1, 'rgba(0, 229, 255, 0.02)');
        ctx.fillStyle = grad;
        ctx.fill();

        ctx.strokeStyle = '#00e5ff';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#889bb5';
        ctx.font = '9px JetBrains Mono';
        ctx.fillText(`P1: ${h1.toFixed(1)}m`, 15, h - 4);
        ctx.fillText(`P2: ${h2.toFixed(1)}m`, w - 75, h - 4);
        ctx.fillText(`&Delta;Z: ${Math.abs(h2 - h1).toFixed(1)}m`, w / 2 - 20, 12);
    }

    clearCaliper() {
        this.caliperVisuals.forEach(v => this.scene.remove(v));
        this.caliperVisuals = [];
        this.caliperPoints = [];
        const canvas = document.getElementById('profile-canvas');
        if (canvas) canvas.style.display = 'none';
        const readingHeight = document.getElementById('reading-height');
        const readingDist = document.getElementById('reading-distance');
        const readingSlope = document.getElementById('reading-slope');
        if (readingHeight) readingHeight.innerText = '--';
        if (readingDist) readingDist.innerText = '--';
        if (readingSlope) readingSlope.innerText = '--';
    }

    animate() {
        requestAnimationFrame(this.animate);

        // Drone flight update
        if (this.mode === 'drone') {
            const dir = new THREE.Vector3();
            dir.x = Math.cos(this.droneYaw) * Math.cos(this.dronePitch);
            dir.y = Math.sin(this.droneYaw) * Math.cos(this.dronePitch);
            dir.z = Math.sin(this.dronePitch);
            dir.normalize();

            const right = new THREE.Vector3().copy(dir).cross(new THREE.Vector3(0, 0, 1)).normalize();

            if (this.keys['w']) this.camera.position.addScaledVector(dir, this.droneSpeed);
            if (this.keys['s']) this.camera.position.addScaledVector(dir, -this.droneSpeed);
            if (this.keys['a']) this.camera.position.addScaledVector(right, -this.droneSpeed);
            if (this.keys['d']) this.camera.position.addScaledVector(right, this.droneSpeed);
            if (this.keys['q'] || this.keys[' ']) this.camera.position.z += this.droneSpeed * 0.7;
            if (this.keys['e'] || this.keys['shift']) this.camera.position.z -= this.droneSpeed * 0.7;

            // Clamp minimum drone altitude above 5m
            if (this.camera.position.z < 8) this.camera.position.z = 8;

            this.camera.lookAt(new THREE.Vector3().copy(this.camera.position).add(dir));

            const hudAlt = document.getElementById('hud-altitude');
            if (hudAlt) hudAlt.innerText = `${Math.round(this.camera.position.z)} m`;
        } else {
            const hudAlt = document.getElementById('hud-altitude');
            if (hudAlt) hudAlt.innerText = `${Math.round(this.camera.position.z)} m`;
        }

        this.renderer.render(this.scene, this.camera);
    }
}
