"""
estimate_elevation.py
─────────────────────
Given a binary segmentation mask and a pre-computed DEM (digital elevation
model / height field), estimate the real-world elevation of the water-body
boundary encoded in the segmentation.

Algorithm
─────────
1. Read Sentinel-2 bands, compute NDWI, and segment water.
2. Extract the contour (border) pixels of the segmentation mask.
3. For every border pixel (y, x), read the DEM elevation  Z[y, x].
4. Compute the gradient magnitude |∇Z| over the entire DEM.
5. Assign a weight to each border pixel:

       w_i = 1 / (|∇Z|[y_i, x_i] + eps)

   Rationale
   ─────────
   • A steep DEM gradient at a border pixel means that many different
     water-surface elevations could produce a contour near that pixel
     (the slope is not informative about elevation).
     → LOW weight.

   • A low DEM gradient (flat terrain) means that the contour uniquely
     identifies a specific elevation.
     → HIGH weight.

6. Return the half sample mode of the weighted elevation values.

Usage
─────
    from estimate_elevation import segment_image, estimate_elevation
    
    seg_mask, _, _ = segment_image("path/to/image.tif", use_DEM=False)
    result = estimate_elevation(DEM, seg_mask)
    print(result["elevation"])                           # predicted (m)
    print(result["std"])                                 # std  (m)
"""

import numpy as np
from scipy.ndimage import sobel, map_coordinates
from typing import Optional
import tifffile
import os
from skimage.morphology import remove_small_objects
from skimage.measure import find_contours
import matplotlib.pyplot as plt

def load_band_idx():
    return {
        "B02": 0,
        "B03": 1,
        "B04": 2,
        "B05": 3,
        "B08": 4,
        "SCL": 5
    }

_SAM_cache = None
_DEM_cache = None

def compute_NDWI(filename):
    # Read full TIFF
    raw = tifffile.imread(filename)
    # Multi-band TIFF is typically loaded as (bands, height, width)
    if len(raw.shape) == 3:
        raw = raw.transpose(1, 2, 0) # transpose to (height, width, bands)
    
    raw = raw.astype(np.float32)
    
    band_idx = load_band_idx()
    B03 = raw[:, :, band_idx["B03"]]
    B08 = raw[:, :, band_idx["B08"]]
    SCL = raw[:, :, band_idx["SCL"]].astype(np.uint8)

    # Cloud mask
    cloud_mask = np.isin(SCL, [1, 3, 8, 9, 10, 11])
    cloud_pct = (np.sum(cloud_mask) / raw.size) * 100.0
    
    # A. NDWI calculation
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (B03 - B08) / (B03 + B08 + 1e-9)
    ndwi = np.nan_to_num(ndwi, nan=-1.0)

    return ndwi, cloud_pct

def segment_image(filename, use_DEM=False, min_area=2000, albufeira="Maranhão"):
    global _SAM_cache, _DEM_cache

    # compute NDWI
    ndwi, cloud_pct = compute_NDWI(filename)
    
    # B. NDWI Thresholding Mask
    ndwi_mask = (ndwi > 0.0) #& ~cloud_mask

    if use_DEM and os.path.exists("generated_data/DEM/{}/SAM.npy".format(albufeira)):
        if _SAM_cache is None:
            _SAM_cache = np.load("generated_data/DEM/{}/SAM.npy".format(albufeira))
        if _DEM_cache is None:
            _DEM_cache = np.load("generated_data/DEM/{}/DEM.npy".format(albufeira))
        acc = _SAM_cache
        DEM = _DEM_cache
        
        # impose bounds on segmentation based on accumulation map
        empty_mask = (acc > .99*np.max(acc))
        full_mask = (acc < np.min(acc))
        ndwi_mask = (ndwi_mask & ~full_mask) | empty_mask
        # estimate elevation and redo segmentation based on DEM
        result = estimate_elevation(DEM, ndwi_mask)
        elevation = result['elevation']
        ndwi_mask = DEM <= elevation
    else:
        # post-process mask to remove outliers (regions with small areas)
        ndwi_mask = remove_small_objects(ndwi_mask, min_size=min_area).astype(np.uint8)
        elevation = 0

    return ndwi_mask, cloud_pct, elevation


# ──────────────────────────────────────────────────────────────────────────────
# Gradient helpers
# ──────────────────────────────────────────────────────────────────────────────

def _dem_gradient_magnitude(DEM: np.ndarray) -> np.ndarray:
    """
    Compute the gradient magnitude of the DEM using Sobel filters.

    Returns an array of the same shape as DEM, with values ≥ 0.
    """
    gx = sobel(DEM, axis=1)
    gy = sobel(DEM, axis=0)
    return np.sqrt(gx**2 + gy**2)


# ──────────────────────────────────────────────────────────────────────────────
# Core estimator
# ──────────────────────────────────────────────────────────────────────────────

def estimate_elevation(
    DEM: np.ndarray,
    seg_mask: np.ndarray,
    *,
    eps: float = 1e-6,
    gradient_magnitude: Optional[np.ndarray] = None,
    return_details: bool = False,
) -> dict:
    """
    Estimate the water-surface elevation from a segmentation mask and a DEM.

    Parameters
    ----------
    DEM : np.ndarray, shape (H, W)
        Reconstructed elevation map in metres.
    seg_mask : np.ndarray, shape (H, W)
        Binary segmentation mask (non-zero = water).
        Accepts both uint8 (0/255) and int/bool (0/1) arrays.
    eps : float
        Small constant added to gradient magnitudes before inverting,
        to avoid division by zero and to cap the maximum weight.
    gradient_magnitude : np.ndarray or None
        Pre-computed |∇Z| of the DEM.  If None it is computed here.
        Pass a cached value when calling this function repeatedly on the
        same DEM to avoid redundant computation.
    return_details : bool
        If True, also return per-pixel arrays (elevations, weights, coords).

    Returns
    -------
    dict with keys
        elevation   – weighted mean elevation estimate (m)
        std         – weighted standard deviation      (m)
        n_pixels    – number of border pixels used
        weights_sum – Σ w_i  (useful for uncertainty comparison across calls)
        details     – only present when return_details=True
            border_ys, border_xs, elevations, weights (all length-N arrays)
    """
    H, W = DEM.shape

    # ── 1. Extract border pixels ───────────────────────────────────────
    # Work directly on the binary mask using morphological erosion to find
    # inner border pixels (water pixels adjacent to land).
    binary_mask = seg_mask > 0

    if not np.any(binary_mask):
        result = dict(elevation=float("nan"), std=float("nan"), n_pixels=0, weights_sum=0.0)
        if return_details:
            result["details"] = dict(border_ys=np.array([]), border_xs=np.array([]),
                                     elevations=np.array([]), weights=np.array([]))
        return result

    # Work on binary mask using skimage find_contours to extract continuous shoreline polygons
    border_ys_list = []
    border_xs_list = []
    
    contours = find_contours(binary_mask.astype(bool), 0.5)
    
    for contour in contours:
        # Swap (row, col) i.e. (y, x) to (x, y) coordinates
        points = contour[:, [1, 0]].astype(np.float64)
        # Skip small noise contours to optimize speed and ignore pixel noise
        if len(points) < 50:
            continue
        
        # Ensure the contour loop is closed
        if not np.allclose(points[0], points[-1]):
            points = np.vstack([points, points[0]])
        
        # Cumulative distance along the contour
        dx = np.diff(points[:, 0])
        dy = np.diff(points[:, 1])
        segment_lengths = np.sqrt(dx**2 + dy**2)
        cumulative_length = np.insert(np.cumsum(segment_lengths), 0, 0.0)
        
        # Remove duplicate cumulative lengths to avoid interpolation issues
        unique_idx = np.insert(np.diff(cumulative_length) > 0, 0, True)
        if not np.any(unique_idx):
            continue
        points = points[unique_idx]
        cumulative_length = cumulative_length[unique_idx]
        
        total_length = cumulative_length[-1]
        if total_length == 0:
            continue
            
        # High-resolution sub-pixel sampling: 5 samples per pixel unit (ds = 0.2)
        sample_step = 0.2
        num_samples = int(np.ceil(total_length / sample_step))
        s_new = np.linspace(0, total_length, num_samples)
        
        xs_sub = np.interp(s_new, cumulative_length, points[:, 0])
        ys_sub = np.interp(s_new, cumulative_length, points[:, 1])
        
        border_xs_list.append(xs_sub)
        border_ys_list.append(ys_sub)
        
    if len(border_ys_list) == 0:
        result = dict(elevation=float("nan"), std=float("nan"), n_pixels=0, weights_sum=0.0)
        if return_details:
            result["details"] = dict(border_ys=np.array([]), border_xs=np.array([]),
                                     elevations=np.array([]), weights=np.array([]))
        return result
        
    border_xs = np.concatenate(border_xs_list)
    border_ys = np.concatenate(border_ys_list)
    
    # Exclude pixels on the image outer boundaries (crop edge rather than true shoreline)
    valid = (border_xs >= 1.0) & (border_xs <= W - 2.0) & (border_ys >= 1.0) & (border_ys <= H - 2.0)
    border_xs = border_xs[valid]
    border_ys = border_ys[valid]

    if len(border_ys) == 0:
        result = dict(elevation=float("nan"), std=float("nan"), n_pixels=0, weights_sum=0.0)
        if return_details:
            result["details"] = dict(border_ys=np.array([]), border_xs=np.array([]),
                                     elevations=np.array([]), weights=np.array([]))
        return result

    # ── 2. Elevation at each border pixel (interpolated DEM values) ──
    elevations = map_coordinates(DEM, [border_ys, border_xs], order=1, mode='nearest').astype(np.float64)

    # ── 3. Gradient magnitude ─────────────────────────────────────────
    if gradient_magnitude is None:
        gradient_magnitude = _dem_gradient_magnitude(DEM)

    # Bilinearly interpolate gradient magnitude at the sub-pixel coordinates
    grad_at_border = map_coordinates(gradient_magnitude, [border_ys, border_xs], order=1, mode='nearest')

    # ── 4. Weights: inversely proportional to gradient magnitude ──────
    #
    #   w_i = 1 / (|∇Z|_i + eps)
    #
    #   • steep slope  → large |∇Z| → small w   (uninformative pixel)
    #   • flat region  → small |∇Z| → large w   (uniquely informative)
    #
    #weights = 1.0 / (grad_at_border + eps)
    weights = 1.0 / (grad_at_border + 1)

    # ── 5. Weighted statistics ────────────────────────────────────────
    w_sum = weights.sum()
    #w_mean = np.dot(weights, elevations) / w_sum
    w_mean = half_sample_mode(elevations)

    # Weighted variance
    w_var = np.dot(weights, (elevations - w_mean) ** 2) / w_sum
    w_std = float(np.sqrt(w_var))

    result = dict(
        elevation=float(w_mean),
        std=w_std,
        n_pixels=len(border_ys),
        weights_sum=float(w_sum),
    )

    if return_details:
        result["details"] = dict(
            border_ys=border_ys,
            border_xs=border_xs,
            elevations=elevations,
            weights=weights,
        )

    return result


def half_sample_mode(data):
    """
    Finds the mode of a continuous, raw dataset using the Half-Sample Mode algorithm.
    Incredibly robust against heavy, long tails.
    """
    # 1. Start by sorting the raw data
    data = np.sort(data)
    
    # 2. Iteratively find the densest half of the data
    while len(data) > 3:
        n = len(data)
        half = n // 2
        
        # Calculate the span (range) of every window containing 50% of the current data
        # The window with the smallest span is the densest region
        spans = data[half:] - data[:-half]
        best_window_idx = np.argmin(spans)
        
        # Keep only the data inside that densest window
        data = data[best_window_idx : best_window_idx + half]
        
    # 3. Return the average of the final remaining points
    return np.mean(data)

# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper: image path → elevation estimate
# ──────────────────────────────────────────────────────────────────────────────

def estimate_elevation_from_image(
    img_path: str,
    DEM: np.ndarray,
    *,
    gradient_magnitude: Optional[np.ndarray] = None,
    eps: float = 1e-6,
    return_details: bool = False,
) -> dict:
    """
    Full pipeline:  satellite image  →  segmentation  →  elevation estimate.

    Parameters
    ----------
    img_path : str
        Path to the satellite / NDWI image.
    DEM : np.ndarray, shape (H, W)
        Pre-computed DEM from terrain_reconstruction.ipynb.
    gradient_magnitude : np.ndarray or None
        Optional cached |∇Z|.  Computed internally if not provided.
    eps, return_details
        Forwarded to estimate_elevation().

    Returns
    -------
    dict – same as estimate_elevation() plus key "seg_mask".
    """
    seg_mask, cloud_pct, _ = segment_image(img_path)

    result = estimate_elevation(
        DEM, seg_mask,
        eps=eps,
        gradient_magnitude=gradient_magnitude,
        return_details=return_details,
    )
    result["seg_mask"] = seg_mask
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ──────────────────────────────────────────────────────────────────────────────

def plot_estimate(DEM: np.ndarray, result: dict, title: str = "Elevation estimate") -> None:
    """
    Quick diagnostic plot.

    Shows the DEM, the border pixels coloured by their weight, and a histogram
    of the elevation distribution at the border (unweighted vs weighted).
    """
    if "details" not in result:
        raise ValueError("Call estimate_elevation(..., return_details=True) first.")

    d = result["details"]
    if len(d["border_ys"]) == 0:
        print("No border pixels – nothing to plot.")
        return

    grad_mag = _dem_gradient_magnitude(DEM)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: DEM with border pixels ──────────────────────────────
    im0 = axes[0].imshow(DEM, cmap="terrain", origin="upper")
    sc  = axes[0].scatter(
        d["border_xs"], d["border_ys"],
        c=d["weights"], cmap="hot", s=1.5, alpha=0.7,
    )
    axes[0].set_title("DEM + border pixels\n(colour = weight)", fontsize=12)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], label="Elevation (m)", fraction=0.046)
    plt.colorbar(sc,  ax=axes[0], label="Weight",        fraction=0.046)

    # ── Panel 2: Gradient magnitude at border ─────────────────────────
    axes[1].scatter(
        d["border_xs"], d["border_ys"],
        c=map_coordinates(grad_mag, [d["border_ys"], d["border_xs"]], order=1, mode='nearest'),
        cmap="inferno", s=1.5, alpha=0.7,
    )
    axes[1].set_xlim(0, DEM.shape[1])
    axes[1].set_ylim(DEM.shape[0], 0)
    axes[1].set_title("|∇DEM| at border pixels\n(high = steep = low weight)", fontsize=12)
    axes[1].set_aspect("equal")

    # ── Panel 3: Weighted elevation histogram ─────────────────────────
    elev = d["elevations"]
    w    = d["weights"] / d["weights"].sum()
    axes[2].hist(elev, bins=60, color="steelblue", alpha=0.5, label="Unweighted")
    axes[2].hist(elev, bins=60, weights=w * len(elev), color="darkorange", alpha=0.6,
                 label="Weighted")
    axes[2].axvline(result["elevation"], color="red", linewidth=2,
                    label=f"Estimate: {result['elevation']:.2f} m")
    axes[2].axvspan(result["elevation"] - result["std"],
                    result["elevation"] + result["std"],
                    alpha=0.15, color="red", label=f"±1σ = {result['std']:.2f} m")
    axes[2].set_xlabel("Elevation (m)")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Border elevation distribution\n({result['n_pixels']:,} pixels)", fontsize=12)
    axes[2].legend(fontsize=8)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Example
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── tiny synthetic demo ──────────────────────────────────────────
    rng = np.random.default_rng(42)
    H, W = 200, 200

    # Synthetic DEM: flat bowl with a gentle slope
    yy, xx = np.mgrid[0:H, 0:W]
    DEM_syn = 120.0 + 0.05 * (xx - W / 2) + 0.02 * (yy - H / 2)
    DEM_syn = DEM_syn.astype(np.float32)

    # Synthetic circular segmentation (water body)
    seg_syn = ((xx - W // 2)**2 + (yy - H // 2)**2 <= 60**2).astype(np.uint8)

    result = estimate_elevation(DEM_syn, seg_syn, return_details=True)

    print("═" * 50)
    print(f"  Estimated elevation : {result['elevation']:.3f} m")
    print(f"  Weighted std        : {result['std']:.3f} m")
    print(f"  Border pixels used  : {result['n_pixels']:,}")
    print("═" * 50)

    plot_estimate(DEM_syn, result, title="Synthetic demo")
