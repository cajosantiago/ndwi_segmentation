# NDWI Water Body Segmentation & Quality Analysis Pipeline

This repository provides an automated pipeline for monitoring reservoir (albufeira) water boundaries, estimating water surface elevations, and rendering spatial water quality index images using Sentinel-2 L2A satellite data and historical water level measurements.

---

## Project Structure

```text
├── README.md                          # Project documentation
├── download_sentinel_data.py          # Automates downloads of Sentinel-2 bands from CDSE
├── estimate_DEM.py                    # Reconstructs Digital Elevation Model (DEM) from historical masks
├── estimate_elevation.py              # Core geometry library (border extraction, elevation estimation)
├── water_segmentation.py              # Main boundary segmentation and level estimation script
├── water_quality.py                   # Computes and renders MAGO water quality index images
│
├── utils/
│   ├── extract_cota_csv.py            # Extracts and cleans historical water level data from Excel
│   ├── plot_elevation.py              # Generates comparative plots of actual vs. predicted elevations
│   └── ...
│
├── docker/
│   └── Dockerfile                     # Containerizes the segmentation pipeline
│
├── data/                              # Input datasets (satellite bands, historical Excel files)
└── generated_data/                    # Output folder (segmentation masks, DEMs, water quality images)
```

---

## End-to-End Guide for a New Albufeira

Follow these 5 steps to configure and run the full monitoring pipeline for a new reservoir/dam.

### Step 1: Download Sentinel-2 L2A Band Data
Use `download_sentinel_data.py` to query the Copernicus Data Space Ecosystem (CDSE) catalog and download required Sentinel-2 bands (`B02`, `B03`, `B04`, `B05`, `B08`, `SCL`) for a specific bounding box (AOI) and date range.

You must provide your SentinelHub API credentials via environment variables (`SH_CLIENT_ID` and `SH_CLIENT_SECRET`) or CLI arguments.

**Command Example:**
```bash
python download_sentinel_data.py \
  --albufeira "Montargil" \
  --aoi-coords -8.1870 39.0389 -8.0549 39.1771 \
  --start-date "2015-01-01" \
  --end-date "2024-12-31" \
  --max-cloud-cover 30 \
  --resolution 10
```
*Outputs: Downloaded multi-band TIFF scenes under `data/sentinelhub/Bandas/Montargil/` and a catalog `index.json` file.*

---

### Step 2: Extract Historical Elevation Data
Extract and format daily water levels ("cota") for the target albufeira from the database Excel file (`data/excel/Historico_2005_2025_V15NOV2025.xlsx`) using `utils/extract_cota_csv.py`.

**Command Example:**
```bash
python utils/extract_cota_csv.py \
  --albufeira "Montargil" \
  --output "data/excel/cota_Montargil.csv"
```
*Outputs: A cleaned, chronological CSV file `data/excel/cota_Montargil.csv` containing columns `date` and `cota`.*

---

### Step 3: Reconstruct the Digital Elevation Model (DEM)
Create a localized high-resolution Digital Elevation Model (DEM) and Sample Accumulation Map (SAM) for the albufeira using `estimate_DEM.py`. 

Before running this, you need a set of initial segmentation masks under `generated_data/segmentation_masks/{albufeira}/ndwi/` to train the optimization. You can generate these initial masks by running `water_segmentation.py` with `use_dem=False` (Step 4 below).

**Command Example:**
```bash
python estimate_DEM.py \
  --albufeira "Montargil" \
  --mask_dir "generated_data/segmentation_masks/Montargil/ndwi" \
  --excel_file "data/excel/cota_Montargil.csv" \
  --iters 3000 \
  --lambda_tv 5.0
```
*Outputs: Reconstructed files `DEM.npy`, `SAM.npy`, and diagnostic plots saved under `generated_data/DEM/Montargil/`.*

---

### Step 4: Run Water Body Segmentation & Elevation Estimation
Segment the reservoir boundaries using NDWI thresholding combined with the reconstructed DEM to improve accuracy (especially in cloudy conditions or shadow areas). Run `water_segmentation.py` and set `--use-dem True`.

**Command Example:**
```bash
python water_segmentation.py \
  --albufeira "Montargil" \
  --use-dem True \
  --save-csv True \
  --cloud-threshold 20.0
```
*Outputs:*
* *Binary segmentation mask `.png` files saved under `generated_data/segmentation_masks/Montargil/ndwi+DEM/`.*
* *Estimated water elevation time-series saved under `data/excel/Montargil/predicted_elevation.csv`.*

To plot and compare the predicted elevations against historical values, run:
```bash
python utils/plot_elevation.py \
  --albufeira "Montargil" \
  --start-date "2021-01-01" \
  --end-date "2023-12-31"
```

---

### Step 5: Run Water Quality Index Analysis
Generate spatial water quality indices (including Chlorophyll-a, Cyanobacteria, Turbidity, CDOM, and TSS) using `water_quality.py`. This script applies the segmentation mask from Step 4 to restrict evaluation to the reservoir boundaries.

**Command Example:**
```bash
python water_quality.py \
  "2023-04-01_2023-05-31" \
  --albufeira "Montargil" \
  --use-dem True \
  --cloud-threshold 20.0 \
  --save-masks
```
*Outputs: Visualized index maps and colorbar-free raw files (`_nb.png`) saved under `generated_data/quality/Montargil/`.*

---

## Running inside Docker

The water segmentation pipeline is containerized for easy deployment. Run the build from the repository root:

```bash
# Build the Docker image
docker build -t water-segmentation -f docker/Dockerfile .

# Run the container
docker run --rm \
  -v $(pwd)/data/sentinelhub/Bandas/Montargil:/data/input \
  -v $(pwd)/generated_data/segmentation_masks/Montargil:/data/output \
  -v $(pwd)/generated_data:/app/generated_data \
  -v $(pwd)/data:/app/data \
  water-segmentation --albufeira "Montargil" --use-dem True
```
