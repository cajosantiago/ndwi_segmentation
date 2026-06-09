import os
import re
import glob
import json
import numpy as np
import pandas as pd
import cv2 as cv
import tifffile
from datetime import datetime

# Prevent Matplotlib from opening GUI windows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Filename pattern: 20230404T113025Z_cc0.2pct_bands.tif
TIF_RE = re.compile(r"^(?P<ts>\d{8}T\d{6}Z)_cc(?P<cc>[\d.]+)pct_bands\.tif$")

def load_band_idx(bands_dir: str):
    index_path = os.path.join(bands_dir, "index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            meta = json.load(f)
        return meta["band_idx"]
    return {"B02": 0, "B03": 1, "B04": 2, "B05": 3, "B08": 4, "SCL": 5}

def main():
    print("=" * 80)
    print("   Cyanobacteria Max Value (cells/mL) Time Series Generator (2022 - 2024)")
    print("=" * 80)

    bands_dir = 'data/Bandas/TWIN_STREAM_Maranhão/input/bands'
    if not os.path.exists(bands_dir):
        raise FileNotFoundError(f"Bands directory not found: {bands_dir}")

    band_idx = load_band_idx(bands_dir)

    # Gather Sentinel-2 multi-band TIFFs for 2022, 2023, and 2024
    tifs = []
    for year in ["2022", "2023", "2024"]:
        tifs.extend(glob.glob(os.path.join(bands_dir, f"{year}*.tif")))
    tifs = sorted(tifs)

    total_files = len(tifs)
    print(f"Found {total_files} TIFF files for years 2022, 2023, and 2024.")

    data_records = []

    for idx, filepath in enumerate(tifs):
        filename = os.path.basename(filepath)
        m = TIF_RE.match(filename)
        if not m:
            continue

        ts_str = m.group("ts")
        cloud_cover = float(m.group("cc"))
        date_str = f"{ts_str[:4]}-{ts_str[4:6]}-{ts_str[6:8]}"
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")

        # Match generate_mago_images.py cloud cover threshold (max 20%)
        if cloud_cover > 20.0:
            print(f"[{idx+1}/{total_files}] Skipping {filename} (Cloud Cover {cloud_cover}% > 20.0%)")
            continue

        print(f"[{idx+1}/{total_files}] Processing: {filename}...")

        try:
            # Read TIFF (transpose to H x W x Bands)
            raw = tifffile.imread(filepath)
            if len(raw.shape) == 3:
                raw = raw.transpose(1, 2, 0)
            raw = raw.astype(np.float32)

            B02 = raw[:, :, band_idx["B02"]]
            B03 = raw[:, :, band_idx["B03"]]
            B04 = raw[:, :, band_idx["B04"]]
            B08 = raw[:, :, band_idx["B08"]]
            SCL = raw[:, :, band_idx["SCL"]].astype(np.uint8)

            # NDWI and Cloud Masks
            with np.errstate(invalid="ignore"):
                NDWI = (B03 - B08) / (B03 + B08 + 1e-9)
            
            # Require blue band reflectance B02 > 0.005 (0.5%) to filter out sensor/atmospheric
            # correction over-correction artifacts (division-by-near-zero peaks at shorelines/shadows)
            water_mask = NDWI >= 0
            cloud_mask = np.isin(SCL, [1, 3, 8, 9, 10, 11])
            valid_mask = water_mask & ~cloud_mask & (B02 > 0.005)

            # Cyanobacteria algorithm (Potes 2018)
            with np.errstate(invalid="ignore", divide="ignore"):
                cyano = 115530.31 * np.power(np.maximum((B03 * B04) / (B02 + 1e-9), 0), 2.38)
            
            cyano[~valid_mask] = np.nan

            # Apply area connected component filter (min 2000 px) to remove small disconnected regions
            valid_mask_img = np.isfinite(cyano)
            binary = (valid_mask_img.astype(np.uint8)) * 255
            n_lbl, lbl_cv, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
            
            clean_mask = np.zeros(binary.shape, dtype=np.uint8)
            for label_idx in range(1, n_lbl):
                if stats[label_idx, cv.CC_STAT_AREA] >= 2000:
                    clean_mask[lbl_cv == label_idx] = 1
            
            cyano[clean_mask == 0] = np.nan

            valid_clean = cyano[np.isfinite(cyano)]
            if len(valid_clean) == 0:
                print(f"  → Date: {date_str} | No valid water body pixels after quality & area filtering.")
                continue

            # Compute maximum and mean values
            max_cyano = float(np.nanmax(valid_clean))
            mean_cyano = float(np.nanmean(valid_clean))

            data_records.append({
                "filename": filename,
                "date": date_str,
                "datetime": date_obj,
                "cloud_pct": cloud_cover,
                "max_cyano_cells_mL": max_cyano,
                "mean_cyano_cells_mL": mean_cyano
            })
            print(f"  → Date: {date_str} | Max Cyano: {max_cyano:,.1f} cells/mL | Mean Cyano: {mean_cyano:,.1f} cells/mL")

        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}: {e}")

    if not data_records:
        print("[ERROR] No data was successfully processed. Exiting.")
        return

    # Convert to DataFrame and sort chronologically
    df = pd.DataFrame(data_records)
    df = df.sort_values(by="datetime").reset_index(drop=True)

    # Save to CSV
    csv_out = "cyano_timeseries_2022_2024.csv"
    df_csv = df.drop(columns=["datetime"])
    df_csv.to_csv(csv_out, index=False)
    print("═" * 80)
    print(f"Successfully saved cyanobacteria time series data to: {csv_out}")
    print("═" * 80)

    # ── Premium Visualisation with WHO Risk Threshold Shading ──
    print("Generating premium cyanobacteria time series plot with primary (Max) and secondary (Mean) axes...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=300)
    ax2 = ax1.twinx()
    
    dates = df["datetime"]
    max_values = df["max_cyano_cells_mL"]
    mean_values = df["mean_cyano_cells_mL"]
    
    # WHO Alert Threshold levels on primary (left) y-axis
    max_y = max(max_values.max() * 1.15, 120000)
    ax1.axhspan(0, 20000, color='#86efac', alpha=0.15, label="WHO Low Risk (<20k)")
    ax1.axhspan(20000, 100000, color='#fef08a', alpha=0.15, label="WHO Moderate Risk (20k - 100k)")
    ax1.axhspan(100000, max_y, color='#fca5a5', alpha=0.15, label="WHO High Risk (>100k)")

    # Plot Maximum values on primary axis (left y-axis, teal)
    line_max = ax1.plot(
        dates, max_values,
        color="#0f766e", linewidth=2.5,
        marker='o', markersize=6, alpha=0.9,
        label="Maximum Cyanobacteria Density", zorder=3
    )
    
    # Plot Mean values on secondary axis (right y-axis, dark orange)
    line_mean = ax2.plot(
        dates, mean_values,
        color="#d97706", linewidth=2.0, linestyle="--",
        marker='s', markersize=5, alpha=0.85,
        label="Mean Cyanobacteria Density", zorder=3
    )
    
    # Format primary axis (left)
    ax1.set_ylim(0, max_y)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax1.set_ylabel("Maximum Cyanobacteria Concentration (cells/mL)", fontsize=12, fontweight="semibold", color="#0f766e", labelpad=12)
    ax1.tick_params(axis='y', colors='#0f766e', labelsize=10)
    
    # Format secondary axis (right)
    ax2.set_ylim(0, mean_values.max() * 1.15)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax2.set_ylabel("Mean Cyanobacteria Concentration (cells/mL)", fontsize=12, fontweight="semibold", color="#d97706", labelpad=12)
    ax2.tick_params(axis='y', colors='#d97706', labelsize=10)
    
    # Grid lines (on primary axis only to avoid visual clutter)
    ax1.grid(True, linestyle="--", alpha=0.5, color="#ccc")
    ax2.grid(False)  # Turn off secondary grid to avoid overlapping grids
    
    # Title & Labels
    ax1.set_title("Reservoir Cyanobacteria Concentration Time Series (2022 - 2024)", fontsize=16, fontweight="bold", pad=20, color="#1e293b")
    ax1.set_xlabel("Observation Date", fontsize=12, fontweight="semibold", labelpad=12, color="#334155")
    
    # Set rotation for dates
    plt.setp(ax1.get_xticklabels(), rotation=15, ha="right")
    ax1.tick_params(axis='x', labelsize=10)
    
    # Combine legends from both axes
    lines = line_max + line_mean
    labels = [l.get_label() for l in lines]
    
    # Add WHO background alerts to the legend list
    who_patches = [
        matplotlib.patches.Patch(color='#86efac', alpha=0.2, label="WHO Low Risk (<20k)"),
        matplotlib.patches.Patch(color='#fef08a', alpha=0.2, label="WHO Moderate Risk (20k - 100k)"),
        matplotlib.patches.Patch(color='#fca5a5', alpha=0.2, label="WHO High Risk (>100k)")
    ]
    
    ax1.legend(handles=who_patches + lines, loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)
    
    # Save high-resolution PNG
    plot_out = "cyano_timeseries_2022_2024.png"
    plt.tight_layout()
    plt.savefig(plot_out, dpi=300)
    print(f"Successfully saved cyanobacteria visualization plot to: {plot_out}")
    print("═" * 80)

if __name__ == "__main__":
    main()
