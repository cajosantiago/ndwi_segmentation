#!/opt/conda/bin/python3.11
"""
generate_mago_images.py

Generates MAGO water quality index images from local Sentinel-2 TIFF band files.
Scans the bands directory directly (no index.json required) and filters by date range.

Usage:
    python generate_mago_images.py 2023-04-01
    python generate_mago_images.py 2023-04-01_2023-05-31
    python generate_mago_images.py 2023-04-01_2023-05-31 --max-cloud 30
    python generate_mago_images.py 2023-04-01_2023-05-31 --albufeira Montargil
    python generate_mago_images.py 2023-04-01_2023-05-31 --out-dir ./my_output
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

# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_ALBUFEIRA     = "Maranhão"
DEFAULT_BASE          = "/home/csantiago/data/Bandas/TWIN_STREAM_Maranhão"
DEFAULT_MAX_CLOUD     = 20.0   # include all by default; user can tighten

# ─── Index definitions (same as notebook) ────────────────────────────────────

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

# Filename pattern: 20230404T113025Z_cc0.2pct_bands.tif
TIF_RE = re.compile(
    r"^(?P<ts>\d{8}T\d{6}Z)_cc(?P<cc>[\d.]+)pct_bands\.tif$"
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_date_arg(arg: str):
    """Return (start_date, end_date) as date objects from '2023-04-01' or '2023-04-01_2023-05-31'."""
    parts = arg.split("_")
    if len(parts) == 2:
        # Could be a date range OR a single date that contains underscores (shouldn't happen)
        # Try parsing as range first
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


def scan_tif_files(bands_dir: str):
    """
    Scan the bands directory for TIF files matching the expected naming convention.
    Returns a list of dicts: {ts, date, cloud_cover, tif_path}
    """
    scenes = []
    for fname in sorted(os.listdir(bands_dir)):
        m = TIF_RE.match(fname)
        if not m:
            continue
        ts_str     = m.group("ts")          # e.g. 20230404T113025Z
        cc         = float(m.group("cc"))   # e.g. 0.2
        date_str   = f"{ts_str[:4]}-{ts_str[4:6]}-{ts_str[6:8]}"
        scenes.append({
            "ts":          ts_str,
            "date":        date_str,
            "cloud_cover": cc,
            "tif":         fname,
        })
    return scenes


def load_band_idx(bands_dir: str):
    """
    Try to load band_idx from index.json; fall back to a hardcoded default
    that matches the standard SentinelHub export order used by the project.
    """
    index_path = os.path.join(bands_dir, "index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            meta = json.load(f)
        return meta["band_idx"]
    # Fallback: standard order from the evalscript (B02 B03 B04 B05 B08 SCL)
    print("[WARN] index.json not found — using default band order: B02=0 B03=1 B04=2 B05=3 B08=4 SCL=5")
    return {"B02": 0, "B03": 1, "B04": 2, "B05": 3, "B08": 4, "SCL": 5}


def compute_indices(tif_path: str, band_idx: dict):
    """Read a TIFF and compute the 8 MAGO water quality indices."""
    raw = tifffile.imread(tif_path).transpose(1, 2, 0).astype(np.float32)

    B02 = raw[:, :, band_idx["B02"]]
    B03 = raw[:, :, band_idx["B03"]]
    B04 = raw[:, :, band_idx["B04"]]
    B05 = raw[:, :, band_idx["B05"]]
    B08 = raw[:, :, band_idx["B08"]]
    SCL = raw[:, :, band_idx["SCL"]].astype(np.uint8)

    # Water / cloud masks
    with np.errstate(invalid="ignore"):
        NDWI = (B03 - B08) / (B03 + B08 + 1e-9)
    water_mask = NDWI >= 0
    cloud_mask = np.isin(SCL, [1, 3, 8, 9, 10, 11])
    valid_mask = water_mask & ~cloud_mask

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
    return all_bands, n_water, NDWI, valid_mask


def save_index_image(band_2d, idx_num, albufeira, date_str, ts_str, cc_tag, out_dir):
    """Apply small-area filter, render and save one index image."""
    mago_colors = ["blue", "cyan", "green", "yellow", "red"]
    mago_cmap   = mcolors.LinearSegmentedColormap.from_list("mago", mago_colors)

    label = INDEX_LABELS[idx_num]
    band  = band_2d.copy().astype(np.float32)

    valid = band[~np.isnan(band)]
    if len(valid) == 0:
        print(f"    [{label}] — sem pixels válidos, ignorado.")
        return

    # Remove small disconnected regions (Guilherme's filter, min 2000 px)
    valid_mask_img = np.isfinite(band)
    binary = (valid_mask_img.astype(np.uint8)) * 255
    n_lbl, lbl_cv, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
    clean_mask = np.zeros(binary.shape, dtype=np.uint8)
    for i in range(1, n_lbl):
        if stats[i, cv.CC_STAT_AREA] >= 2000:
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
        "--base",
        default=DEFAULT_BASE,
        help=f"Base directory containing input/bands/ (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=DEFAULT_MAX_CLOUD,
        dest="max_cloud",
        help=f"Maximum cloud cover %% to include (default: {DEFAULT_MAX_CLOUD} = all)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        dest="out_dir",
        help="Output directory for images (default: MAGO_<albufeira>_imgs/)",
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
    print(f"\n{'═'*60}")
    print(f"  MAGO image generator")
    print(f"  Albufeira : {args.albufeira}")
    print(f"  Intervalo : {start_str} → {end_str}")
    print(f"  Cloud max : {args.max_cloud}%")
    print(f"  Índices   : {[INDEX_LABELS[i] for i in INDICES_TO_COMPUTE]}")
    print(f"  Guardar máscaras (NDWI, valid_mask): {'sim' if args.save_masks else 'não'}")
    print(f"{'═'*60}\n")

    # ── Locate bands directory ────────────────────────────────────────────────
    bands_dir = os.path.join(args.base, "input", "bands")
    if not os.path.isdir(bands_dir):
        print(f"[ERROR] Bands directory not found: {bands_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load band index ───────────────────────────────────────────────────────
    band_idx = load_band_idx(bands_dir)

    # ── Scan all TIF files in the folder ─────────────────────────────────────
    all_scenes = scan_tif_files(bands_dir)
    print(f"TIF files found in folder : {len(all_scenes)}")

    # ── Filter by date range and cloud cover ──────────────────────────────────
    scenes = [
        s for s in all_scenes
        if start_str <= s["date"] <= end_str
        and s["cloud_cover"] <= args.max_cloud
    ]
    print(f"Cenas no intervalo (cloud ≤ {args.max_cloud}%) : {len(scenes)}\n")

    if not scenes:
        print("Nenhuma cena encontrada no intervalo especificado.")
        sys.exit(0)

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = args.out_dir or f"MAGO_{args.albufeira}_imgs"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Imagens guardadas em: {out_dir}\n")

    # ── Process each scene ────────────────────────────────────────────────────
    for scene_idx, scene in enumerate(scenes, start=1):
        date_str    = scene["date"]
        ts_str      = scene["ts"]
        cloud_cover = scene["cloud_cover"]
        tif_path    = os.path.join(bands_dir, scene["tif"])
        cc_tag      = f"cc{cloud_cover:.1f}pct"

        print(f"[{scene_idx:03d}/{len(scenes)}] {date_str}  (cloud={cloud_cover:.1f}%)", end=" ... ")

        if not os.path.exists(tif_path):
            print("TIFF não encontrado — ignorado.")
            continue

        try:
            all_bands, n_water, ndwi, valid_mask = compute_indices(tif_path, band_idx)
        except Exception as exc:
            print(f"ERRO ao ler TIFF: {exc}")
            continue

        if n_water == 0:
            print("0 pixels de água válidos — cena ignorada.")
            continue

        print(f"OK ({n_water} px água)")

        if args.save_masks:
            save_ndwi_image(ndwi, args.albufeira, date_str, ts_str, cc_tag, out_dir)
            save_valid_mask_image(valid_mask, args.albufeira, date_str, ts_str, cc_tag, out_dir)

        for idx_num in INDICES_TO_COMPUTE:
            band_2d = all_bands[:, :, idx_num]
            save_index_image(band_2d, idx_num, args.albufeira,
                             date_str, ts_str, cc_tag, out_dir)

    print(f"\nConcluído. Imagens guardadas em: {out_dir}")


if __name__ == "__main__":
    main()
