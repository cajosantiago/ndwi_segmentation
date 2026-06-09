import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime

# Prevent Matplotlib from opening GUI windows (headless mode)
# This also ensures segment_image's plt.show() does not block execution!
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from estimate_elevation import segment_image, estimate_elevation

def main():
    print("=" * 70)
    print("   Sentinel-2 Water Area Time Series Generator (2025 - 2026)")
    print("=" * 70)

    bands_dir = 'data/Bandas/TWIN_STREAM_Maranhão/input/bands'
    if not os.path.exists(bands_dir):
        raise FileNotFoundError(f"Bands directory not found: {bands_dir}")

    # Find all Sentinel-2 multi-band TIFFs for 2025 and 2026
    tifs_25 = glob.glob(os.path.join(bands_dir, "2025*.tif"))
    tifs_26 = glob.glob(os.path.join(bands_dir, "2026*.tif"))
    tif_paths = sorted(tifs_25 + tifs_26)

    total_files = len(tif_paths)
    print(f"Found {total_files} TIFF files matching years 2025 and 2026.")

    data_records = []

    for idx, filepath in enumerate(tif_paths):
        filename = os.path.basename(filepath)
        print(f"[{idx+1}/{total_files}] Processing: {filename}...")

        # Parse date from filename (format: YYYYMMDDThhmmssZ...)
        try:
            date_str = filename.split('T')[0]
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            formatted_date = date_obj.strftime("%Y-%m-%d")
        except Exception as e:
            print(f"  [WARN] Failed to parse date from filename: {filename}. Skipping.")
            continue

        try:
            # Run segmentation (plt.show is bypassed due to Agg backend)
            ndwi_mask, cloud_pct, _ = segment_image(filepath)

            # Estimate elevation
            DEM = np.load("DEM.npy")
            result = estimate_elevation(DEM, ndwi_mask)
            ndwi_mask = DEM <= result['elevation']+.01
            
            # Area calculation: 1 pixel of Sentinel-2 = 10m x 10m = 100 m2
            pixel_count = int(np.sum(ndwi_mask > 0))
            area_m2 = pixel_count * 100
            area_ha = area_m2 / 10000.0  # Hectares
            area_km2 = area_m2 / 1e6      # Square Kilometers

            data_records.append({
                "filename": filename,
                "date": formatted_date,
                "datetime": date_obj,
                "cloud_pct": float(cloud_pct),
                "water_pixels": pixel_count,
                "area_m2": area_m2,
                "area_ha": area_ha,
                "area_km2": area_km2
            })
            print(f"  → Date: {formatted_date} | Area: {area_km2:.3f} km² ({area_ha:.1f} ha) | Clouds: {cloud_pct:.1f}%")
        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}: {e}")

    if not data_records:
        print("[ERROR] No data was successfully processed. Exiting.")
        return

    # Convert to pandas DataFrame and sort by date
    df = pd.DataFrame(data_records)
    df = df.sort_values(by="datetime").reset_index(drop=True)

    # Save to CSV
    csv_out = "area_time_series_2025_2026.csv"
    # Drop datetime object column for clean CSV output
    df_csv = df.drop(columns=["datetime"])
    df_csv.to_csv(csv_out, index=False)
    print("═" * 70)
    print(f"Successfully saved time series data to: {csv_out}")
    print("═" * 70)

    # ── Premium Visualisation ──
    print("Generating premium time series plot...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    
    # Enable anti-aliasing and sleek grid
    ax.grid(True, linestyle="--", alpha=0.5, color="#ccc")
    
    dates = df["datetime"]
    areas = df["area_km2"]
    clouds = df["cloud_pct"]
    
    # Sleek area line
    ax.plot(dates, areas, color="#2b5c8f", linewidth=2.5, alpha=0.8, label="Water Area (km²)", zorder=1)
    
    # Scatter points colored by cloud percentage
    # Premium coolwarm or viridis colormap representing cloud cover
    sc = ax.scatter(
        dates, areas,
        c=clouds, cmap="plasma",
        s=80, edgecolors="#1a365d", linewidths=1.5,
        alpha=0.9, zorder=2, label="Observation"
    )
    
    # Colorbar styling
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Cloud Cover (%)", fontsize=11, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    
    # Title & Labels
    ax.set_title("Water-Body Surface Area Time Series (2025 - 2026)", fontsize=16, fontweight="bold", pad=20, color="#1a202c")
    ax.set_xlabel("Observation Date", fontsize=12, fontweight="semibold", labelpad=12, color="#2d3748")
    ax.set_ylabel("Water Area (km²)", fontsize=12, fontweight="semibold", labelpad=12, color="#2d3748")
    
    # Layout adjustment and spacing
    plt.xticks(rotation=15)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Legend
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=10)
    
    # Save high-resolution PNG
    plot_out = "area_time_series_2025_2026.png"
    plt.tight_layout()
    plt.savefig(plot_out, dpi=300)
    print(f"Successfully saved visualization plot to: {plot_out}")
    print("═" * 70)

if __name__ == "__main__":
    main()
