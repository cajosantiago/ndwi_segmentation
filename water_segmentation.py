import os
import re
import sys
import argparse
import json
import numpy as np
import tifffile
from PIL import Image
from estimate_elevation import segment_image
import pandas as pd

# Filename pattern: 20230404T113025Z_cc0.2pct_bands.tif
TIF_RE = re.compile(
    r"^(?P<ts>\d{8}T\d{6}Z)_cc(?P<cc>[\d.]+)pct_bands\.tif$"
)

def scan_tif_files(bands_dir):
    """
    Scan the bands directory for TIF files matching the expected naming convention.
    Returns a list of dicts: {ts, date, filename_cloud, tif_path, filename}
    """
    scenes = []
    if not os.path.exists(bands_dir):
        return scenes
    for fname in sorted(os.listdir(bands_dir)):
        m = TIF_RE.match(fname)
        if not m:
            continue
        ts_str = m.group("ts")          # e.g. 20230404T113025Z
        cc = float(m.group("cc"))       # e.g. 0.2
        date_str = f"{ts_str[:4]}-{ts_str[4:6]}-{ts_str[6:8]}"
        scenes.append({
            "ts": ts_str,
            "date": date_str,
            "filename_cloud": cc,
            "filename": fname,
            "tif_path": os.path.join(bands_dir, fname)
        })
    return scenes

def load_band_idx(bands_dir):
    return {"B02": 0, "B03": 1, "B04": 2, "B05": 3, "B08": 4, "SCL": 5}

def main():
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Water Body Segmentation using NDWI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--albufeira",
        default="Maranhão",
        help="Chose name of albufeira"
    )
    parser.add_argument(
        "--input-dir",
        default="/home/csantiago/data/sentinelhub/Bandas/Maranhão",
        help="Input folder with bands TIFF files"
    )
    parser.add_argument(
        "--output-dir",
        default="/home/csantiago/generated_data/segmentation_masks/Maranhão/ndwi+dem_teste",
        help="Output folder for saving masks and previews"
    )
    parser.add_argument(
        "--cloud-threshold",
        type=float,
        default=20.0,
        help="Maximum cloud coverage percentage allowed"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=2000,
        help="Minimum pixel area for NDWI mask post-processing"
    )
    parser.add_argument(
        "--filter",
        type=bool,
        default=False,
        help="Apply a 3-point rolling median filter to remove transient outliers (e.g. single overpass segmentation errors)"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("  WATER SEGMENTATION PIPELINE (NDWI)")
    print("="*80)
    print(f"  Input Directory   : {args.input_dir}")
    print(f"  Output Directory  : {args.output_dir}")
    print("="*80 + "\n")
    
    # 1. Scan input folder
    scenes = scan_tif_files(args.input_dir)
    if not scenes:
        print(f"[ERROR] No valid TIFF files matching pattern found in: {args.input_dir}")
        sys.exit(1)
        
    print(f"Discovered {len(scenes)} total TIFF files in input folder.")
    
    # Load band indices
    band_idx = load_band_idx(args.input_dir)
    if "B03" not in band_idx or "B08" not in band_idx or "SCL" not in band_idx:
        print("[ERROR] Required bands (B03, B08, SCL) not found in index.json. Aborting.")
        sys.exit(1)
        
    # 2. Compare cloud coverage and filter scenes
    filtered_scenes = []
    print("\n--- Cloud Cover Comparison and Filtering Analysis ---")
    print(f"{'Date':12s} | {'Filename':35s} | {'Global (File)%':15s} | {'Local (SCL)%':12s} | {'Decision':10s}")
    print("-" * 95)
    
    for s in scenes:
        # Read only SCL band to quickly compute cloud coverage without loading full TIFF
        try:
            raw_scl = tifffile.imread(s["tif_path"], key=band_idx["SCL"]).astype(np.uint8)
            cloud_mask = np.isin(raw_scl, [1, 3, 8, 9, 10, 11])
            scl_cloud_pct = (np.sum(cloud_mask) / raw_scl.size) * 100.0
        except Exception as e:
            print(f"Error reading SCL band for {s['filename']}: {e}")
            scl_cloud_pct = 100.0 # assume fully cloudy on error
            cloud_mask = np.ones((1,1), dtype=bool)
            
        s["local_cloud"] = scl_cloud_pct
        if scl_cloud_pct <= args.cloud_threshold:
            decision = "INCLUDE"
            filtered_scenes.append(s)
        else:
            decision = "EXCLUDE"
            
        print(f"{s['date']:12s} | {s['filename'][:35]:35s} | {s['filename_cloud']:14.2f}% | {s['local_cloud']:11.2f}% | {decision:10s}")
        
    print("-" * 95)
    print(f"Total scenes included: {len(filtered_scenes)} / {len(scenes)}\n")
    
    if not filtered_scenes:
        print("No scenes remaining after cloud filtering. Exiting.")
        sys.exit(0)
        
    elevation = []
    # 4. Process each scene
    for idx, s in enumerate(filtered_scenes, 1):
        print(f"[{idx}/{len(filtered_scenes)}] Processing {s['date']} ({s['filename']})...")
        
        # NDWI segmentation
        ndwi_mask, _, result = segment_image(s['tif_path'], albufeira=args.albufeira, use_DEM=True)
        elevation.append((s['date'], result['elevation']))

        # Save mask
        ndwi_mask_path = os.path.join(args.output_dir, f"{s['filename']}.png")
        Image.fromarray((ndwi_mask * 255).astype(np.uint8)).save(ndwi_mask_path)
        print(f"  → Saved NDWI mask to: {os.path.basename(ndwi_mask_path)}")
            
    print("\n" + "="*80)
    print("  PIPELINE PROCESSING COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")

    # save cota do csv
    df_p = pd.DataFrame(elevation, columns=["date", "elevation"])

    if args.filter and len(df_p) > 3:
        # Apply a 3-point rolling median filter to remove transient outliers (e.g. single overpass segmentation errors)
        df_p['elevation'] = df_p['elevation'].rolling(window=3, center=True, min_periods=1).median()

        # Preserve values at the edges (first and last elements remain unfiltered)
        if len(df_p) > 0:
            df_p.loc[df_p.index[0], 'elevation'] = df_p['elevation'].iloc[0]
        if len(df_p) > 1:
            df_p.loc[df_p.index[-1], 'elevation'] = df_p['elevation'].iloc[-1]

    # Save csv timeseries
    df_p.to_csv(os.path.join(args.output_dir, "predicted_elevation.csv"), index=False)
    print("  → Saved elevation data to: {}".format(os.path.join(args.output_dir, "predicted_elevation.csv")))

if __name__ == "__main__":
    main()
