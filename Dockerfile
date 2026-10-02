# DepthWizard — ISRO SAC SIH26175
# Single-View Height Estimation & 3D Flythrough System
# Production Docker Image

FROM python:3.11-slim

# System dependencies for OpenCV, rasterio, and GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \n    libexpat1 \
    libgeos-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-download Depth Anything V2 model weights during build
RUN python -c "from transformers import AutoImageProcessor, AutoModelForDepthEstimation; \
    AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf'); \
    AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')" \
    || echo "Model pre-download skipped (will download on first inference)"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["uvicorn", "src.depth_wizard.server:app", "--host", "0.0.0.0", "--port", "8000"]
