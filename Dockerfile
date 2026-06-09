# ------------------------------------------------------------------
#  Docker image to run the `water_segmentation.py` pipeline.
#
#  * The image is based on python:3.11-slim (minimal, Debian-based).
#    This allows using pre-built binary wheels for numpy, scipy, etc.,
#    avoiding lengthy compilation times.
#  * Runs as a non-root user (UID 10001) for container security.
#  * Expected volume mounts:
#        /data/input           - Sentinel-2 TIFF scenes
#        /data/output          - Generated segmentation masks
#        /app/generated_data   - Optional: DEM data (SAM.npy/DEM.npy files)
#        /app/data             - Optional: Output predicted elevation CSVs
#
#  Build example:
#        docker build -t water-segmentation -f docker/Dockerfile docker/
#
#  Run example (using local directories):
#        docker run --rm \
#          -v $(pwd)/data/sentinelhub/Bandas/Maranhão:/data/input \
#          -v $(pwd)/generated_data/segmentation_masks/Maranhão:/data/output \
#          -v $(pwd)/generated_data:/app/generated_data \
#          -v $(pwd)/data:/app/data \
#          water-segmentation --albufeira "Maranhão"
#
#  Note on permissions:
#    If you encounter permission issues with mounted volumes, you can run
#    the container with your host user's UID and GID:
#        docker run --user $(id -u):$(id -g) ...
# ------------------------------------------------------------------

# Use a slim, stable Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer streams for real-time logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Create a non-root user and group for security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

# Install Python dependencies
# We keep the pip cache clean and combine installs to minimize layer count
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pillow \
    scikit-image \
    pandas \
    tifffile

# Copy the Python scripts and assign ownership to the non-root user
COPY --chown=appuser:appgroup water_segmentation.py estimate_elevation.py ./
COPY --chown=appuser:appgroup ./DEM/DEM.npy ./DEM/SAM.npy ./

# Switch to the non-root user
USER appuser

# Define default entrypoint to run the pipeline script
ENTRYPOINT ["python", "water_segmentation.py"]

# Default parameters (can be overridden at runtime)
CMD ["--input-dir", "/data/input", "--output-dir", "/data/output"]

# Metadata labels
LABEL org.opencontainers.image.title="Water-Segmentation" \
      org.opencontainers.image.description="Lightweight container to run the water segmentation pipeline using NDWI and DEM analysis." \
      org.opencontainers.image.authors="CS+MM"
