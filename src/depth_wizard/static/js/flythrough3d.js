/**
 * DepthWizard 3D Flythrough Engine
 * =================================
 * Three.js WebGL rendering engine providing:
 * - 3D terrain surface reconstruction from single-view satellite DSM
 * - Optical satellite texture mapping
 * - First-Person Drone Flight controls (WASD + Mouse) & Orbit mode
 * - 3D Laser Measurement Caliper (rooftop height & distance)
 * - Dynamic Flood Inundation Water Plane
 * - Directional Sun Angle & Shadow Simulation
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
        this.ambientLight = null;
        
        // Flight state
        this.mode = 'orbit'; // 'orbit' or 'drone'
        this.keys = {};
        this.mouseState = { isDown: false, prevX: 0, prevY: 0 };
        this.cameraRotation = { yaw: 0, pitch: -0.5 };
        this.droneSpeed = 1.2;

        // Laser caliper state
        this.caliperActive = false;
        this.currentExaggeration = 1.0;
        this.caliperPoints = [];
        this.caliperVisuals = [];
        this.raycaster = new THREE.Raycaster();
        this.mouseVec = new THREE.Vector2();

        // Terrain metadata cache
        this.currentPayload = null;

        this.init();
    }

    init() {
        const width = this.container.clientWidth || window.innerWidth;
        const height = this.container.clientHeight || (window.innerHeight - 56);

        // 1. Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x06090e);
        this.scene.fog = new THREE.FogExp2(0x06090e, 0.003);

        // 2. Camera
        this.camera = new THREE.PerspectiveCamera(60, width / height, 0.5, 2000);
        this.camera.position.set(0, -120, 90);
        this.camera.up.set(0, 0, 1); // Z is UP (elevation)
        this.camera.lookAt(0, 0, 15);

        // 3. Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // 4. Lighting
        this.ambientLight = new THREE.AmbientLight(0xffffff, 0.45);
        this.scene.add(this.ambientLight);

        this.sunLight = new THREE.DirectionalLight(0xffffff, 0.95);
        this.sunLight.position.set(100, -80, 150);
        this.sunLight.castShadow = true;
        this.sunLight.shadow.mapSize.width = 1024;
        this.sunLight.shadow.mapSize.height = 1024;
        this.scene.add(this.sunLight);

        // 5. Water Plane (for flood simulation)
        const waterGeo = new THREE.PlaneGeometry(250, 250);
        const waterMat = new THREE.MeshStandardMaterial({
            color: 0x0099ff,
            transparent: true,
            opacity: 0.55,
            roughness: 0.1,
            metalness: 0.8
        });
        this.waterMesh = new THREE.Mesh(waterGeo, waterMat);
        this.waterMesh.position.z = -10; // Hidden initially
        this.scene.add(this.waterMesh);

        // 6. Event Listeners
        this.bindEvents();

        // 7. Render Loop
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
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
        this.renderer.domElement.addEventListener('mousedown', (e) => {
            this.mouseState.isDown = true;
            this.mouseState.prevX = e.clientX;
            this.mouseState.prevY = e.clientY;

            // If caliper tool is active, handle click
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
                this.cameraRotation.yaw -= dx * 0.004;
                this.cameraRotation.pitch = Math.max(-1.4, Math.min(1.4, this.cameraRotation.pitch - dy * 0.004));
            } else {
                // Orbit mode
                const radius = this.camera.position.distanceTo(new THREE.Vector3(0, 0, 10));
                this.cameraRotation.yaw -= dx * 0.008;
                this.cameraRotation.pitch = Math.max(0.1, Math.min(1.4, this.cameraRotation.pitch + dy * 0.008));
                
                this.camera.position.x = radius * Math.cos(this.cameraRotation.yaw) * Math.cos(this.cameraRotation.pitch);
                this.camera.position.y = radius * Math.sin(this.cameraRotation.yaw) * Math.cos(this.cameraRotation.pitch);
                this.camera.position.z = radius * Math.sin(this.cameraRotation.pitch);
                this.camera.lookAt(0, 0, 15);
            }
        });

        // Mouse wheel for zoom in Orbit mode
        this.renderer.domElement.addEventListener('wheel', (e) => {
            if (this.mode === 'orbit') {
                const forward = new THREE.Vector3();
                this.camera.getWorldDirection(forward);
                this.camera.position.addScaledVector(forward, -e.deltaY * 0.15);
            }
        });
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
            this.terrainMesh.material.dispose();
            this.terrainMesh = null;
        }

        // Clear caliper visuals
        this.clearCaliper();

        // 1. Create Plane Geometry (Z is elevation)
        // Mesh spans 160 x 160 units in WebGL space
        const planeDim = 160;
        const geometry = new THREE.PlaneGeometry(planeDim, planeDim, gridS - 1, gridS - 1);
        const positionAttr = geometry.attributes.position;

        // 2. Displace Vertices according to estimated DSM
        // Note: PlaneGeometry is created in XY plane. We map normalized_z to Z coordinate.
        const zExaggeration = Math.min(45.0, Math.max(12.0, (elevRange / 100.0) * 35.0));
        
        for (let i = 0; i < positionAttr.count; i++) {
            const zNorm = normalizedZ[i] || 0.0;
            // Elevate vertex via raw typed array
            positionAttr.array[i * 3 + 2] = zNorm * zExaggeration;
        }
        positionAttr.needsUpdate = true;
        geometry.computeVertexNormals();

        // 3. Precompute Hypsometric Elevation & Slope Attributes
        const count = positionAttr.count;
        const elevColors = new Float32Array(count * 3);
        const slopeColors = new Float32Array(count * 3);
        const normals = geometry.attributes.normal;

        for (let i = 0; i < count; i++) {
            const zNorm = normalizedZ[i] || 0.0;
            // Hypsometric tint: Blue -> Cyan -> Green -> Yellow -> Red
            let r = 0, g = 0, b = 0;
            if (zNorm < 0.25) {
                const t = zNorm / 0.25;
                r = 0.05; g = 0.2 + 0.6 * t; b = 0.9;
            } else if (zNorm < 0.5) {
                const t = (zNorm - 0.25) / 0.25;
                r = 0.05 + 0.3 * t; g = 0.8 + 0.15 * t; b = 0.9 - 0.8 * t;
            } else if (zNorm < 0.75) {
                const t = (zNorm - 0.5) / 0.25;
                r = 0.35 + 0.6 * t; g = 0.95 - 0.2 * t; b = 0.1;
            } else {
                const t = (zNorm - 0.75) / 0.25;
                r = 0.95; g = 0.75 - 0.65 * t; b = 0.1 + 0.3 * t;
            }
            elevColors[i * 3] = r;
            elevColors[i * 3 + 1] = g;
            elevColors[i * 3 + 2] = b;

            // Slope angle relative to vertical [0, 0, 1]
            const nz = normals ? (normals.array[i * 3 + 2] || 1.0) : 1.0;
            const slopeDeg = Math.acos(Math.max(-1, Math.min(1, nz))) * (180 / Math.PI);
            if (slopeDeg < 15) {
                slopeColors[i * 3] = 0.1; slopeColors[i * 3 + 1] = 0.85; slopeColors[i * 3 + 2] = 0.35;
            } else if (slopeDeg < 35) {
                slopeColors[i * 3] = 0.95; slopeColors[i * 3 + 1] = 0.75; slopeColors[i * 3 + 2] = 0.1;
            } else {
                slopeColors[i * 3] = 0.95; slopeColors[i * 3 + 1] = 0.2; slopeColors[i * 3 + 2] = 0.15;
            }
        }

        this.elevColorAttr = new THREE.BufferAttribute(elevColors, 3);
        this.slopeColorAttr = new THREE.BufferAttribute(slopeColors, 3);
        this.vertexColorMaterial = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.6,
            metalness: 0.1,
            side: THREE.DoubleSide
        });

        // 4. Texture Loading from Base64 Data URL
        const textureLoader = new THREE.TextureLoader();
        textureLoader.load(payload.texture_data_url, (tex) => {
            tex.wrapS = THREE.ClampToEdgeWrapping;
            tex.wrapT = THREE.ClampToEdgeWrapping;
            tex.generateMipmaps = true;

            this.opticalMaterial = new THREE.MeshStandardMaterial({
                map: tex,
                roughness: 0.75,
                metalness: 0.1,
                flatShading: false,
                side: THREE.DoubleSide
            });

            this.terrainMesh = new THREE.Mesh(geometry, this.opticalMaterial);
            this.terrainMesh.receiveShadow = true;
            this.terrainMesh.castShadow = true;
            this.scene.add(this.terrainMesh);

            // Center camera
            this.camera.position.set(0, -110, 85);
            this.camera.lookAt(0, 0, 15);
            this.cameraRotation.yaw = -Math.PI / 2;
            this.cameraRotation.pitch = 0.6;
        });

        // Set default water level below ground
        this.waterMesh.position.z = -10;
    }

    setShadingMode(mode) {
        if (!this.terrainMesh) return;
        if (mode === 'optical') {
            if (this.opticalMaterial) this.terrainMesh.material = this.opticalMaterial;
        } else if (mode === 'elevation') {
            this.terrainMesh.geometry.setAttribute('color', this.elevColorAttr);
            this.terrainMesh.material = this.vertexColorMaterial;
        } else if (mode === 'slope') {
            this.terrainMesh.geometry.setAttribute('color', this.slopeColorAttr);
            this.terrainMesh.material = this.vertexColorMaterial;
        }
    }

    setCameraView(viewName) {
        if (viewName === 'nadir') {
            this.camera.position.set(0, 0, 150);
            this.camera.lookAt(0, 0, 0);
            this.cameraRotation.pitch = 1.5;
            this.setMode('orbit');
        } else if (viewName === 'oblique') {
            this.camera.position.set(0, -110, 85);
            this.camera.lookAt(0, 0, 15);
            this.cameraRotation.yaw = -Math.PI / 2;
            this.cameraRotation.pitch = 0.6;
            this.setMode('orbit');
        } else if (viewName === 'fpv') {
            this.camera.position.set(0, -45, 20);
            this.camera.lookAt(0, 0, 18);
            this.setMode('drone');
        }
    }

    setMode(newMode) {
        this.mode = newMode;
        if (newMode === 'drone') {
            document.getElementById('hud-mode-text').innerText = 'DRONE FLYTHROUGH (WASD + MOUSE)';
            document.getElementById('crosshair').style.display = 'block';
        } else {
            document.getElementById('hud-mode-text').innerText = 'ORBIT INSPECTION (ROTATE & ZOOM)';
            document.getElementById('crosshair').style.display = 'none';
        }
    }

    setCaliperTool(active) {
        this.caliperActive = active;
        const btn = document.getElementById('btn-caliper');
        if (active) {
            btn.classList.add('active');
            this.clearCaliper();
            document.getElementById('caliper-status').innerText = 'CLICK ON ROOFTOP / POINT 1';
        } else {
            btn.classList.remove('active');
            document.getElementById('caliper-status').innerText = 'READY';
        }
    }

    
    handleHover(e) {
        if (!this.terrainMesh || !this.currentPayload) return;
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouseVec.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouseVec.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouseVec, this.camera);
        const intersects = this.raycaster.intersectObject(this.terrainMesh);
        const probe = document.getElementById('hud-probe');
        if (intersects.length > 0 && probe) {
            const pt = intersects[0].point;
            const elevRange = this.currentPayload.elevation_range_m || 50.0;
            const minElev = this.currentPayload.min_elevation_m || 0.0;
            const zExag = (this.currentExaggeration || 1.0) * Math.min(45.0, Math.max(12.0, (elevRange / 100.0) * 35.0));
            const metricZ = minElev + Math.max(0, (pt.z / Math.max(0.1, zExag)) * elevRange);
            probe.innerText = 'CURSOR: ' + metricZ.toFixed(1) + 'm ASL';
        }
    }

    setExaggeration(factor) {
        if (!this.terrainMesh) return;
        this.currentExaggeration = factor;
        this.terrainMesh.scale.z = factor;
    }

    toggleWireframe() {
        if (!this.terrainMesh || !this.terrainMesh.material) return false;
        this.terrainMesh.material.wireframe = !this.terrainMesh.material.wireframe;
        return this.terrainMesh.material.wireframe;
    }

    drawElevationProfile(p1, p2) {
        const canvas = document.getElementById('profile-canvas');
        if (!canvas || !this.currentPayload) return;
        canvas.style.display = 'block';
        const ctx = canvas.getContext('2d');
        const W = canvas.width;
        const H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        const elevRange = this.currentPayload.elevation_range_m || 50.0;
        const minElev = this.currentPayload.min_elevation_m || 0.0;
        const zExag = (this.currentExaggeration || 1.0) * Math.min(45.0, Math.max(12.0, (elevRange / 100.0) * 35.0));

        const samples = [];
        const numSamples = 40;
        for (let i = 0; i <= numSamples; i++) {
            const t = i / numSamples;
            const testPt = new THREE.Vector3(
                p1.x * (1 - t) + p2.x * t,
                p1.y * (1 - t) + p2.y * t,
                200
            );
            const ray = new THREE.Raycaster(testPt, new THREE.Vector3(0, 0, -1));
            const hits = ray.intersectObject(this.terrainMesh);
            if (hits.length > 0) {
                const zM = minElev + (hits[0].point.z / Math.max(0.1, zExag)) * elevRange;
                samples.push(zM);
            } else {
                samples.push(minElev);
            }
        }

        const minS = Math.min(...samples);
        const maxS = Math.max(...samples);
        const span = Math.max(0.5, maxS - minS);

        // Gradient Area
        const grad = ctx.createLinearGradient(0, 0, 0, H);
        grad.addColorStop(0, 'rgba(0, 229, 255, 0.35)');
        grad.addColorStop(1, 'rgba(0, 229, 255, 0.02)');

        ctx.beginPath();
        ctx.moveTo(10, H - 12);
        for (let i = 0; i < samples.length; i++) {
            const x = 10 + (i / (samples.length - 1)) * (W - 20);
            const y = (H - 14) - ((samples[i] - minS) / span) * (H - 28);
            ctx.lineTo(x, y);
        }
        ctx.lineTo(W - 10, H - 12);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();

        // Stroke line
        ctx.beginPath();
        for (let i = 0; i < samples.length; i++) {
            const x = 10 + (i / (samples.length - 1)) * (W - 20);
            const y = (H - 14) - ((samples[i] - minS) / span) * (H - 28);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = '#00e5ff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Labels
        ctx.fillStyle = '#00e676';
        ctx.font = '9px monospace';
        ctx.fillText('Peak: ' + maxS.toFixed(1) + 'm', 12, 12);
        ctx.fillStyle = '#94a3b8';
        ctx.fillText('Base: ' + minS.toFixed(1) + 'm', W - 78, H - 4);
    }

    takeSnapshot() {
        this.renderer.render(this.scene, this.camera);
        const dataUrl = this.renderer.domElement.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = 'DepthWizard_ISRO_SAC_3D_' + Date.now() + '.png';
        link.href = dataUrl;
        link.click();
    }

    handleCaliperClick(e) {
        if (!this.terrainMesh) return;
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
            const markerMat = new THREE.MeshBasicMaterial({ 
                color: this.caliperPoints.length === 0 ? 0xff1744 : 0x00e676 
            });
            const marker = new THREE.Mesh(markerGeo, markerMat);
            marker.position.copy(pt);
            this.scene.add(marker);
            this.caliperVisuals.push(marker);

            this.caliperPoints.push(pt);

            if (this.caliperPoints.length === 1) {
                document.getElementById('caliper-status').innerText = 'CLICK ON GROUND / POINT 2';
            } else if (this.caliperPoints.length === 2) {
                const p1 = this.caliperPoints[0];
                const p2 = this.caliperPoints[1];

                // Draw glowing laser line connecting points
                const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
                const lineMat = new THREE.LineBasicMaterial({ color: 0x00e5ff, linewidth: 3 });
                const line = new THREE.Line(lineGeo, lineMat);
                this.scene.add(line);
                this.caliperVisuals.push(line);

                // Compute metric distance based on terrain scaling
                const elevRange = this.currentPayload ? this.currentPayload.elevation_range_m : 50.0;
                const zExaggeration = Math.min(45.0, Math.max(12.0, (elevRange / 100.0) * 35.0));
                
                // Convert WebGL units back to real-world meters
                const deltaZ_webgl = Math.abs(p2.z - p1.z);
                const metricHeightM = (deltaZ_webgl / zExaggeration) * elevRange;
                const metricDistM = p1.distanceTo(p2) * (160.0 / 256.0) * 0.6;

                // Update UI readouts
                document.getElementById('reading-height').innerText = metricHeightM.toFixed(1) + ' m';
                document.getElementById('reading-distance').innerText = metricDistM.toFixed(1) + ' m';
                document.getElementById('reading-slope').innerText = ((deltaZ_webgl / Math.max(0.1, p1.distanceTo(new THREE.Vector3(p2.x, p2.y, p1.z)))) * 100).toFixed(0) + ' %';
                document.getElementById('caliper-status').innerText = 'MEASUREMENT COMPLETED';
                this.drawElevationProfile(p1, p2);
            }
        }
    }

    clearCaliper() {
        this.caliperVisuals.forEach(v => this.scene.remove(v));
        this.caliperVisuals = [];
        this.caliperPoints = [];
        const canvas = document.getElementById('profile-canvas');
        if (canvas) canvas.style.display = 'none';
    }

    setWaterLevel(levelNormalized) {
        if (!this.waterMesh || !this.currentPayload) return;
        const elevRange = this.currentPayload.elevation_range_m;
        const zExaggeration = Math.min(45.0, Math.max(12.0, (elevRange / 100.0) * 35.0));
        
        if (levelNormalized <= 0.01) {
            this.waterMesh.position.z = -10; // Submerged
        } else {
            this.waterMesh.position.z = levelNormalized * zExaggeration;
        }
    }

    setSunAngle(angleDeg) {
        if (!this.sunLight) return;
        const rad = (angleDeg * Math.PI) / 180;
        this.sunLight.position.x = 160 * Math.cos(rad);
        this.sunLight.position.z = 160 * Math.sin(rad);
    }


    renderLandingZones(zones) {
        this.clearLandingZones();
        if (!zones || zones.length === 0 || !this.currentPayload) return;

        const elevRange = this.currentPayload.elevation_range_m;
        const zExaggeration = Math.min(45.0, Math.max(12.0, (elevRange / 100.0) * 35.0));
        const terrainScale = 200.0;

        zones.forEach(z => {
            const group = new THREE.Group();
            
            // Outer glowing landing ring
            const ringGeo = new THREE.RingGeometry(3.0, 4.0, 32);
            const ringMat = new THREE.MeshBasicMaterial({ color: 0x00e676, side: THREE.DoubleSide });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            group.add(ring);

            // Inner beacon cylinder
            const cylGeo = new THREE.CylinderGeometry(0.4, 0.4, 8, 16);
            const cylMat = new THREE.MeshBasicMaterial({ color: 0x00e676 });
            const beacon = new THREE.Mesh(cylGeo, cylMat);
            beacon.rotation.x = Math.PI / 2;
            beacon.position.z = 4;
            group.add(beacon);

            // Position in terrain space
            const worldX = (z.norm_x - 0.5) * terrainScale;
            const worldY = (z.norm_y - 0.5) * terrainScale;
            const minZ = this.currentPayload.min_elevation_m;
            const normZ = (z.elevation_m - minZ) / (elevRange + 1e-6);
            const worldZ = normZ * zExaggeration + 0.5;

            group.position.set(worldX, worldY, worldZ);
            this.scene.add(group);
            if (!this.landingZoneMarkers) this.landingZoneMarkers = [];
            this.landingZoneMarkers.push(group);
        });
    }

    clearLandingZones() {
        if (this.landingZoneMarkers) {
            this.landingZoneMarkers.forEach(m => this.scene.remove(m));
            this.landingZoneMarkers = [];
        }
    }

    animate() {
        requestAnimationFrame(this.animate);

        // Drone flight update
        if (this.mode === 'drone') {
            const dir = new THREE.Vector3();
            const forward = new THREE.Vector3(
                Math.cos(this.cameraRotation.yaw) * Math.cos(this.cameraRotation.pitch),
                Math.sin(this.cameraRotation.yaw) * Math.cos(this.cameraRotation.pitch),
                Math.sin(this.cameraRotation.pitch)
            ).normalize();

            const right = new THREE.Vector3(-Math.sin(this.cameraRotation.yaw), Math.cos(this.cameraRotation.yaw), 0).normalize();
            const up = new THREE.Vector3(0, 0, 1);

            if (this.keys['w']) this.camera.position.addScaledVector(forward, this.droneSpeed);
            if (this.keys['s']) this.camera.position.addScaledVector(forward, -this.droneSpeed);
            if (this.keys['a']) this.camera.position.addScaledVector(right, -this.droneSpeed);
            if (this.keys['d']) this.camera.position.addScaledVector(right, this.droneSpeed);
            if (this.keys[' ']) this.camera.position.addScaledVector(up, this.droneSpeed);
            if (this.keys['shift']) this.camera.position.addScaledVector(up, -this.droneSpeed);

            this.camera.lookAt(this.camera.position.clone().add(forward));

            // Update altitude telemetry
            const altElem = document.getElementById('hud-altitude');
            if (altElem) {
                altElem.innerText = Math.max(0, Math.round(this.camera.position.z * 1.5)) + ' m';
            }
        }

        // Render
        this.renderer.render(this.scene, this.camera);
    }
}
