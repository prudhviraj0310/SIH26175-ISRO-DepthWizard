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
            // Elevate vertex
            positionAttr.setZ(i, zNorm * zExaggeration);
        }
        geometry.computeVertexNormals();

        // 3. Texture Loading from Base64 Data URL
        const textureLoader = new THREE.TextureLoader();
        textureLoader.load(payload.texture_data_url, (tex) => {
            tex.wrapS = THREE.ClampToEdgeWrapping;
            tex.wrapT = THREE.ClampToEdgeWrapping;
            tex.generateMipmaps = true;

            const material = new THREE.MeshStandardMaterial({
                map: tex,
                roughness: 0.75,
                metalness: 0.1,
                flatShading: false,
                side: THREE.DoubleSide
            });

            this.terrainMesh = new THREE.Mesh(geometry, material);
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
            }
        }
    }

    clearCaliper() {
        this.caliperVisuals.forEach(v => this.scene.remove(v));
        this.caliperVisuals = [];
        this.caliperPoints = [];
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
