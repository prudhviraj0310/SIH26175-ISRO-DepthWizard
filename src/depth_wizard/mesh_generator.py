"""
DepthWizard 3D Mesh Generator
==============================
Transforms 2D Digital Surface Models (DSM) and optical satellite imagery
into 3D textured terrain meshes ready for Three.js WebGL flythrough rendering.
"""

import io
import base64
import numpy as np
from PIL import Image
from typing import Dict, Any

class MeshGenerator:
    def __init__(self, target_grid_size: int = 128):
        self.target_grid_size = target_grid_size

    def generate_mesh_payload(
        self,
        dsm: np.ndarray,
        rgb_image: np.ndarray,
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Processes DSM and RGB into a compact WebGL payload:
        - Downsamples grid to target_grid_size x target_grid_size for smooth 60 FPS Three.js rendering
        - Computes normalized height values and true metric heights
        - Encodes the optical satellite texture as a Base64 JPEG
        - Computes surface normal vectors for high-precision sunlight shading
        """
        orig_rows, orig_cols = dsm.shape
        target_s = self.target_grid_size

        # Subsample DSM to target grid size
        row_indices = np.linspace(0, orig_rows - 1, target_s).astype(int)
        col_indices = np.linspace(0, orig_cols - 1, target_s).astype(int)
        sub_dsm = dsm[np.ix_(row_indices, col_indices)]

        # Resize RGB texture to match high-resolution texture map (512x512)
        pil_img = Image.fromarray(rgb_image)
        pil_img = pil_img.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Save to Base64 JPEG buffer
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        texture_base64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        # Elevation bounds
        min_z = float(np.min(dsm))
        max_z = float(np.max(dsm))
        z_range = max(1.0, max_z - min_z)

        # Normalized Z in [0, 1] for WebGL plane displacement
        normalized_z = (sub_dsm - min_z) / z_range

        # Compute surface normals for realistic terrain shading
        dz_dx = np.gradient(sub_dsm, axis=1)
        dz_dy = np.gradient(sub_dsm, axis=0)
        # Normal vector N = (-dz/dx, -dz/dy, 1) normalized
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(sub_dsm) * 2.0
        norm_mag = np.sqrt(nx**2 + ny**2 + nz**2)
        nx /= norm_mag
        ny /= norm_mag
        nz /= norm_mag

        # Convert to flat lists for JSON serialization
        heights_flat = [round(float(h), 2) for h in sub_dsm.flatten()]
        norm_z_flat = [round(float(z), 4) for z in normalized_z.flatten()]

        return {
            "grid_size": target_s,
            "min_elevation_m": round(min_z, 2),
            "max_elevation_m": round(max_z, 2),
            "elevation_range_m": round(z_range, 2),
            "metric_heights": heights_flat,
            "normalized_z": norm_z_flat,
            "texture_data_url": texture_base64,
            "stats": stats
        }
