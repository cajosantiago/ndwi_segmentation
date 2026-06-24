#!/opt/conda/bin/python3.11
"""
water_quality.py

Generates MAGO water quality index images from local Sentinel-2 TIFF band files.
Scans the bands directory directly (no index.json required) and filters by date range.

Usage:
    python water_quality.py 2023-04-01
    python water_quality.py 2023-04-01_2023-05-31
    python water_quality.py 2023-04-01_2023-05-31 --max-cloud 30
    python water_quality.py 2023-04-01_2023-05-31 --albufeira Montargil
    python water_quality.py 2023-04-01_2023-05-31 --out-dir ./my_output
"""

import os
import re
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2 as cv
import tifffile
from datetime import datetime, date
from PIL import Image
from estimate_elevation import segment_image
from water_segmentation import scan_tif_files, load_band_idx

# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_ALBUFEIRA     = "Maranhão"
DEFAULT_CLOUD_THRESH  = 20.0   # maximum cloud cover % allowed
DEFAULT_MIN_AREA      = 2000

# ─── Index definitions ────────────────────────────────────────────────────────

INDICES_TO_COMPUTE = [0, 3, 5, 6, 7]

INDEX_LABELS = {
    0: "chla_ndci_mg_m3",
    1: "chla_high_mg_m3",
    2: "chla_low_mg_m3",
    3: "cyano_cells_mL",
    4: "cyano_mg_m3",
    5: "turbidity_NTU",
    6: "CDOM_ug_L",
    7: "TSS_mg_L",
}

INDEX_PRETTY = {
    0: "Clorofila-a [mg/m³] (NDCI, Mishra 2012)",
    1: "Clorofila-a [mg/m³] (Soria-Perpinyà 2021, altos)",
    2: "Clorofila-a [mg/m³] (Soria-Perpinyà 2021, baixos)",
    3: "Cianobactérias [células/mL] (Potes 2018)",
    4: "Cianobactérias [mg/m³] (Soria-Perpinyà 2021)",
    5: "Turbidez [NTU] (Zhan 2022)",
    6: "CDOM [µg/L] (Soria-Perpinyà 2021)",
    7: "TSS [mg/L] (Soria-Perpinyà 2021)",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_date_arg(arg: str):
    """Return (start_date, end_date) as date objects from '2023-04-01' or '2023-04-01_2023-05-31'."""
    parts = arg.split("_")
    if len(parts) == 2:
        try:
            start = datetime.strptime(parts[0], "%Y-%m-%d").date()
            end   = datetime.strptime(parts[1], "%Y-%m-%d").date()
            return start, end
        except ValueError:
            pass
    # Single date
    try:
        d = datetime.strptime(arg, "%Y-%m-%d").date()
        return d, d
    except ValueError:
        raise ValueError(
            f"Cannot parse date argument '{arg}'. "
            "Expected 'YYYY-MM-DD' or 'YYYY-MM-DD_YYYY-MM-DD'."
        )


def compute_indices(tif_path: str, band_idx: dict, ndwi_mask: np.ndarray):
    """Read a TIFF and compute the 8 MAGO water quality indices using the provided ndwi_mask."""
    raw = tifffile.imread(tif_path)
    if len(raw.shape) == 3:
        raw = raw.transpose(1, 2, 0)
    raw = raw.astype(np.float32)

    B02 = raw[:, :, band_idx["B02"]]
    B03 = raw[:, :, band_idx["B03"]]
    B04 = raw[:, :, band_idx["B04"]]
    B05 = raw[:, :, band_idx["B05"]]
    B08 = raw[:, :, band_idx["B08"]]
    SCL = raw[:, :, band_idx["SCL"]].astype(np.uint8)

    # Cloud mask
    cloud_mask = np.isin(SCL, [1, 3, 8, 9, 10, 11])
    
    # Combined valid mask (water mask from segment_image and cloud-free)
    valid_mask = (ndwi_mask > 0) & ~cloud_mask

    def make_band(arr):
        b = arr.copy().astype(np.float32)
        b[~valid_mask] = np.nan
        return b

    with np.errstate(invalid="ignore", divide="ignore"):
        NDCI    = (B05 - B04) / (B05 + B04 + 1e-9)
        i0 = make_band(14.039 + 86.11 * NDCI + 194.325 * NDCI ** 2)
        i1 = make_band(19.866 * np.power(np.maximum(B05 / (B04 + 1e-9), 0), 2.3051))
        ratio02 = np.maximum(B03, B02) / (B03 + 1e-9)
        i2 = make_band(np.power(10, -2.4792 * np.log10(np.maximum(ratio02, 1e-9)) - 0.0389))
        i3 = make_band(115530.31 * np.power(np.maximum((B03 * B04) / (B02 + 1e-9), 0), 2.38))
        i4 = make_band(21.554 * np.power(np.maximum(B05 / (B04 + 1e-9), 0), 3.47941))
        Rrs_red = B04 / np.pi
        Rrs_nir = B08 / np.pi
        T_low   = 228.1  * Rrs_red / (1.0 - Rrs_red / 0.1641 + 1e-9)
        T_high  = 3078.9 * Rrs_nir / (1.0 - Rrs_nir / 0.2112 + 1e-9)
        w       = np.clip((Rrs_red - 0.025) / 0.005, 0, 1)
        i5 = make_band(np.maximum((1.0 - w) * T_low + w * T_high, 0))
        i6 = make_band(2.4072 * (B04 / (B02 + 1e-9)) + 0.0709)
        i7 = make_band(355.85 * Rrs_red / (1.0 - Rrs_red / 0.1904 + 1e-9))

    all_bands = np.stack([i0, i1, i2, i3, i4, i5, i6, i7], axis=-1)
    n_water   = int(np.sum(valid_mask))
    
    # Calculate NDWI for return
    with np.errstate(invalid="ignore", divide="ignore"):
        NDWI = (B03 - B08) / (B03 + B08 + 1e-9)
    NDWI = np.nan_to_num(NDWI, nan=-1.0)
    
    return all_bands, n_water, NDWI, valid_mask


def save_index_image(band_2d, idx_num, albufeira, date_str, ts_str, cc_tag, out_dir, min_area=2000):
    """Apply small-area filter, render and save one index image."""
    mago_colors = ["blue", "cyan", "green", "yellow", "red"]
    mago_cmap   = mcolors.LinearSegmentedColormap.from_list("mago", mago_colors)

    label = INDEX_LABELS[idx_num]
    band  = band_2d.copy().astype(np.float32)

    valid = band[~np.isnan(band)]
    if len(valid) == 0:
        print(f"    [{label}] — sem pixels válidos, ignorado.")
        return

    # Remove small disconnected regions (Guilherme's filter)
    valid_mask_img = np.isfinite(band)
    binary = (valid_mask_img.astype(np.uint8)) * 255
    n_lbl, lbl_cv, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
    clean_mask = np.zeros(binary.shape, dtype=np.uint8)
    for i in range(1, n_lbl):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            clean_mask[lbl_cv == i] = 1
    band[clean_mask == 0] = np.nan

    valid_clean = band[np.isfinite(band)]
    if len(valid_clean) == 0:
        print(f"    [{label}] — todos os pixels removidos pelo filtro de área, ignorado.")
        return

    if label == "cyano_cells_mL":
        vmin = 0.0
        vmax = 1500.0
    else:
        vmin = np.nanpercentile(valid_clean, 2)
        vmax = np.nanpercentile(valid_clean, 98)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(band, cmap=mago_cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=INDEX_PRETTY[idx_num])
    ax.set_title(f"{albufeira} — {date_str}\n{INDEX_PRETTY[idx_num]}")
    ax.axis("off")

    fname = os.path.join(out_dir, f"{ts_str}_{cc_tag}_{label}.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    → {os.path.basename(fname)}")

    #save band without colorbar
    fname_nb = os.path.join(out_dir, f"{ts_str}_{cc_tag}_{label}_nb.png")
    plt.imsave(fname_nb, band, cmap=mago_cmap, vmin=vmin, vmax=vmax)
    print(f"    → {os.path.basename(fname_nb)}")


def save_ndwi_image(ndwi, albufeira, date_str, ts_str, cc_tag, out_dir):
    """Save NDWI as a diverging blue-white-red image (range -1..1, 0-contour shown)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(ndwi, cmap="RdBu", vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax, label="NDWI")
    # Mark the NDWI=0 threshold on the colourbar
    cbar.ax.axhline(0.5, color="black", linewidth=1.2, linestyle="--")
    cbar.ax.text(1.05, 0.5, "0", transform=cbar.ax.transAxes,
                 va="center", ha="left", fontsize=9)
    ax.set_title(f"{albufeira} — {date_str}\nNDWI  (B03−B08)/(B03+B08)")
    ax.axis("off")
    fname = os.path.join(out_dir, f"{ts_str}_{cc_tag}_NDWI.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    → {os.path.basename(fname)}")


def save_valid_mask_image(valid_mask, albufeira, date_str, ts_str, cc_tag, out_dir):
    """Save valid_mask (water & cloud-free pixels) as a binary black/cyan image."""
    from matplotlib.colors import ListedColormap
    cmap_bin = ListedColormap(["black", "cyan"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(valid_mask.astype(np.uint8), cmap=cmap_bin, vmin=0, vmax=1)
    n_valid = int(valid_mask.sum())
    ax.set_title(
        f"{albufeira} — {date_str}\n"
        f"Máscara válida (água & sem nuvem)  —  {n_valid:,} px"
    )
    ax.axis("off")
    fname = os.path.join(out_dir, f"{ts_str}_{cc_tag}_valid_mask.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    → {os.path.basename(fname)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate MAGO water quality images from local Sentinel-2 TIF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "date_range",
        help=(
            "Single date 'YYYY-MM-DD' or range 'YYYY-MM-DD_YYYY-MM-DD'. "
            "Example: 2023-04-01_2023-05-31"
        ),
    )
    parser.add_argument(
        "--albufeira",
        default=DEFAULT_ALBUFEIRA,
        help=f"Reservoir name used in titles and output folder (default: {DEFAULT_ALBUFEIRA})",
    )
    parser.add_argument(
        "--use-dem",
        type=bool,
        default=False,
        help="Use DEM to improve segmentation"
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Input folder with bands TIFF files"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for images (default: generated_data/quality/<albufeira>/)",
    )
    parser.add_argument(
        "--cloud-threshold",
        type=float,
        default=DEFAULT_CLOUD_THRESH,
        help=f"Maximum cloud cover %% to include (default: {DEFAULT_CLOUD_THRESH}%)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help=f"Minimum pixel area for NDWI mask post-processing (default: {DEFAULT_MIN_AREA})",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        default=False,
        dest="save_masks",
        help="Also save NDWI and valid_mask images for each scene.",
    )
    args = parser.parse_args()

    # ── Parse dates ──────────────────────────────────────────────────────────
    try:
        start_date, end_date = parse_date_arg(args.date_range)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    # Set up default paths if not provided
    if args.input_dir is None:
        args.input_dir = f"/home/csantiago/data/sentinelhub/Bandas/{args.albufeira}"
    
    if args.output_dir is None:
        args.output_dir = f"/home/csantiago/generated_data/quality/{args.albufeira}"

    print(f"\n{'═'*80}")
    print(f"  MAGO WATER QUALITY INDEX PIPELINE")
    print(f"{'═'*80}")
    print(f"  Albufeira  : {args.albufeira}")
    print(f"  Interval   : {start_str} → {end_str}")
    print(f"  Cloud Thresh: {args.cloud_threshold}%")
    print(f"  Min Area   : {args.min_area} px")
    print(f"  Use DEM    : {args.use_dem}")
    print(f"  Input Dir  : {args.input_dir}")
    print(f"  Output Dir : {args.output_dir}")
    print(f"  Save Masks : {'yes' if args.save_masks else 'no'}")
    print(f"{'═'*80}\n")

    # ── Verify and scan bands directory ───────────────────────────────────────
    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load band index ───────────────────────────────────────────────────────
    band_idx = load_band_idx(args.input_dir)
    if "B03" not in band_idx or "B08" not in band_idx or "SCL" not in band_idx:
        print("[ERROR] Required bands (B03, B08, SCL) not found. Aborting.", file=sys.stderr)
        sys.exit(1)

    # ── Scan all TIF files in the folder ─────────────────────────────────────
    scenes = scan_tif_files(args.input_dir)
    if not scenes:
        print(f"[ERROR] No valid TIFF files matching pattern found in: {args.input_dir}")
        sys.exit(1)

    print(f"Discovered {len(scenes)} total TIFF files in input folder.")

    # ── Filter by date range first ────────────────────────────────────────────
    date_filtered_scenes = [
        s for s in scenes
        if start_str <= s["date"] <= end_str
    ]
    print(f"Scenes within date range ({start_str} to {end_str}): {len(date_filtered_scenes)}")

    if not date_filtered_scenes:
        print("No scenes found in the specified date range.")
        sys.exit(0)

    # ── Filter by cloud cover (Local SCL) ─────────────────────────────────────
    filtered_scenes = []
    print("\n--- Cloud Cover Comparison and Filtering Analysis ---")
    print(f"{'Date':12s} | {'Filename':35s} | {'Global (File)%':15s} | {'Local (SCL)%':12s} | {'Decision':10s}")
    print("-" * 95)
    
    for s in date_filtered_scenes:
        try:
            raw_scl = tifffile.imread(s["tif_path"], key=band_idx["SCL"]).astype(np.uint8)
            cloud_mask = np.isin(raw_scl, [1, 3, 8, 9, 10, 11])
            scl_cloud_pct = (np.sum(cloud_mask) / raw_scl.size) * 100.0
        except Exception as e:
            print(f"Error reading SCL band for {s['filename']}: {e}")
            scl_cloud_pct = 100.0
            
        s["local_cloud"] = scl_cloud_pct
        if scl_cloud_pct <= args.cloud_threshold:
            decision = "INCLUDE"
            filtered_scenes.append(s)
        else:
            decision = "EXCLUDE"
            
        print(f"{s['date']:12s} | {s['filename'][:35]:35s} | {s['filename_cloud']:14.2f}% | {s['local_cloud']:11.2f}% | {decision:10s}")
        
    print("-" * 95)
    print(f"Total scenes included: {len(filtered_scenes)} / {len(date_filtered_scenes)}\n")
    
    if not filtered_scenes:
        print("No scenes remaining after cloud filtering. Exiting.")
        sys.exit(0)

    # ── Output directory ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Images will be saved to: {args.output_dir}\n")

    # ── Process each scene ────────────────────────────────────────────────────
    for scene_idx, s in enumerate(filtered_scenes, start=1):
        date_str    = s["date"]
        ts_str      = s["ts"]
        cloud_cover = s["local_cloud"]
        tif_path    = s["tif_path"]
        cc_tag      = f"cc{cloud_cover:.1f}pct"

        print(f"[{scene_idx:03d}/{len(filtered_scenes)}] Processing {date_str} (cloud={cloud_cover:.1f}%)...")

        try:
            # 1. Run the segmentation pipeline to obtain the water mask (using reuse of segment_image)
            ndwi_mask, _, result = segment_image(
                tif_path, 
                albufeira=args.albufeira, 
                use_DEM=args.use_dem, 
                min_area=args.min_area
            )
            
            # 2. Compute the water quality indices using the segmented mask
            all_bands, n_water, ndwi, valid_mask = compute_indices(tif_path, band_idx, ndwi_mask)
        except Exception as exc:
            print(f"  [ERROR] Processing scene: {exc}")
            continue

        if n_water == 0:
            print("  → 0 pixels of valid water — scene ignored.")
            continue

        print(f"  → OK ({n_water} px water)")

        if args.save_masks:
            save_ndwi_image(ndwi, args.albufeira, date_str, ts_str, cc_tag, args.output_dir)
            save_valid_mask_image(valid_mask, args.albufeira, date_str, ts_str, cc_tag, args.output_dir)

        for idx_num in INDICES_TO_COMPUTE:
            band_2d = all_bands[:, :, idx_num]
            save_index_image(band_2d, idx_num, args.albufeira,
                             date_str, ts_str, cc_tag, args.output_dir, min_area=args.min_area)

    print(f"\nConcluído. Imagens guardadas em: {args.output_dir}")


if __name__ == "__main__":
    main()
