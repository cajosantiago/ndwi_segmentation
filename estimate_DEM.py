import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
import os
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim

def total_variation(Z):
    """Anisotropic total variation: mean of absolute differences between neighbouring pixels."""
    diff_y = torch.abs(Z[1:, :] - Z[:-1, :])   # vertical neighbours
    diff_x = torch.abs(Z[:, 1:] - Z[:, :-1])   # horizontal neighbours
    return torch.mean(diff_y) + torch.mean(diff_x)

def reconstruct_terrain(
    H, W,
    border_ys, border_xs, border_heights,
    accumulation,
    lambda_tv=5.0,
    n_iters=3000,
    lr=0.5,
    device=None,
    verbose=True,
    log_interval=500,
):
    """
    Reconstruct a dense Digital Elevation Model (DEM) from contour observations
    using PyTorch gradient descent, subject to hard boundary constraints.

    The terrain Z (shape H x W) is optimized to minimise:

        E(Z) = L_contour(Z) + lambda_tv * L_tv(Z)

    where:
        L_contour  = mean((Z[y_i, x_i] - h_i)^2)   contour consistency
        L_tv       = anisotropic total variation      smoothness regularizer

    Boundary Hard Constraints:
        - Z[accumulation == 0] = max(border_heights)
        - Z[accumulation == max_accumulation] = min(border_heights)
    These constraints are enforced exactly at each step, representing a
    projected gradient descent rather than a regularization term.

    Parameters
    ----------
    H, W : int
        Height and width of the terrain grid.
    border_ys : array-like, shape (N,)
        Row indices of all segmentation boundary pixels.
    border_xs : array-like, shape (N,)
        Column indices of all segmentation boundary pixels.
    border_heights : array-like, shape (N,)
        Known real-world elevation at each boundary pixel.
    accumulation : np.ndarray, shape (H, W)
        Water accumulation count map.
    lambda_tv : float
        Weight for the total variation regularization term.
    n_iters : int
        Number of gradient descent iterations.
    lr : float
        Learning rate for the Adam optimizer.
    device : str or torch.device, optional
        Compute device ('cpu' or 'cuda'). Auto-detected if None.
    verbose : bool
        If True, print loss every `log_interval` iterations.
    log_interval : int
        Logging frequency.

    Returns
    -------
    Z_np : np.ndarray, shape (H, W)
        Reconstructed elevation map in metres.
    loss_history : list of dict
        Per-logged-iteration dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Device : {device}")
    print(f"Grid   : {H} x {W}  ({H*W:,} pixels)")
    print(f"Obs    : {len(border_ys):,} boundary pixels")
    print(f"lambda_tv={lambda_tv}, lr={lr}, n_iters={n_iters}")
    print("-" * 60)

    # ── Boundary levels from observation heights ──────────────────────
    min_elevation = float(np.min(border_heights))
    max_elevation = float(np.max(border_heights))

    # ── Set up hard constraints tensors ───────────────────────────────
    accum_tensor = torch.tensor(accumulation, dtype=torch.float32, device=device)
    zero_accum_mask = (accum_tensor == 0.0)
    max_accum_mask = (accum_tensor == accum_tensor.max())

    print(f"Hard constraints: zero accumulation = {max_elevation:.2f} m ({torch.sum(zero_accum_mask).item():,} px)")
    print(f"                 max accumulation  = {min_elevation:.2f} m ({torch.sum(max_accum_mask).item():,} px)")

    # ── Initialise learnable terrain ──────────────────────────────────
    # Start from the mean observed height so the optimiser has a warm start.
    h_mean = float(np.mean(border_heights))
    Z = torch.full((H, W), h_mean, dtype=torch.float32, device=device, requires_grad=True)

    # Initialize constrained pixels to their target values
    with torch.no_grad():
        Z[zero_accum_mask] = max_elevation
        Z[max_accum_mask] = min_elevation

    optimizer = optim.Adam([Z], lr=lr)

    # ── Observation tensors ───────────────────────────────────────────
    ys      = torch.tensor(border_ys,      dtype=torch.long,    device=device)
    xs      = torch.tensor(border_xs,      dtype=torch.long,    device=device)
    heights = torch.tensor(border_heights, dtype=torch.float32, device=device)

    loss_history = []

    # ── Optimisation loop ─────────────────────────────────────────────
    for it in range(n_iters):
        optimizer.zero_grad()

        # 1) Contour consistency: minimise squared error at boundary pixels
        z_at_contour = Z[ys, xs]
        L_contour = torch.mean((z_at_contour - heights) ** 2)

        # 2) Total variation: penalise large local elevation gradients
        L_tv = total_variation(Z)

        # 3) Total energy
        loss = L_contour + lambda_tv * L_tv

        loss.backward()
        optimizer.step()

        # Enforce hard constraints (Projected Gradient Descent step)
        with torch.no_grad():
            Z[zero_accum_mask] = max_elevation
            Z[max_accum_mask] = min_elevation

        if verbose and (it % log_interval == 0 or it == n_iters - 1):
            entry = {
                "iter":      it,
                "loss":      loss.item(),
                "L_contour": L_contour.item(),
                "L_tv":      L_tv.item(),
            }
            loss_history.append(entry)
            print(
                f"  iter {it:5d}/{n_iters-1}  "
                f"loss={entry['loss']:.4f}  "
                f"L_contour={entry['L_contour']:.4f}  "
                f"L_tv={entry['L_tv']:.4f}"
            )

    Z_np = Z.detach().cpu().numpy()
    return Z_np, loss_history

def process_mask(mask_path):
    """
    Load a water segmentation mask and return:
        binary   – (H, W) int32 mask (1 = water, 0 = land)
        border   – (N, 2) int32 array of coordinates [row, col] along the boundary
    """
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask at {mask_path}")

    # binary mask (1 for water, 0 for land)
    binary = (mask > 127).astype(np.int32)

    # To find contours, we need a uint8 mask of 0/255
    contours, _ = cv.findContours((binary * 255).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)

    if not contours:
        return binary, np.empty((0, 2), dtype=np.int32)

    # Use the largest contour
    largest_contour = max(contours, key=cv.contourArea)
    pts = largest_contour.squeeze()
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]

    # cv2 contours are in [col, row] -> swap to [row, col] / [y, x]
    border_pixels = pts[:, [1, 0]].astype(np.int32)
    return binary, border_pixels

def to_dt(s):
    if isinstance(s, datetime):
        return s
    if hasattr(s, 'to_pydatetime'):
        return s.to_pydatetime()
    s = str(s).strip()
    return datetime.strptime(s[:10], "%Y-%m-%d")

# ── Main Script Entry ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    input_folder = "data/segmentation_masks"

    # get masks filenames from input_folder
    mask_paths = sorted(
        str(mask_file)
        for mask_file in Path(input_folder).glob("*.png")
    )
    print(f"Found {len(mask_paths)} mask files in {input_folder}.")

    df_excel = pd.read_excel('data/excel/AlbufeirasMaranhao_18-07-2025.xlsx')
    print(f"Loaded Excel file with {len(df_excel)} rows.")

    # Parse metadata from mask file names
    results = []
    for img_path in mask_paths:
        filename = os.path.basename(img_path)
        # Discard segmentations from 2025 and 2026
        year = filename[0:4]
        if year in ["2025", "2026"]:
            continue
        # Extract date from prefix (YYYYMMDD)
        date_str = f"{year}-{filename[4:6]}-{filename[6:8]}"
        # Extract cloud cover percentage
        try:
            cloud_per = float(filename.split('_cc')[1].split('pct')[0])
        except Exception:
            cloud_per = 0.0
        results.append((img_path, date_str, cloud_per))

    df_images = pd.DataFrame(results, columns=["path", "date", "cloud"])
    print(f"Using {len(df_images)} mask files after discarding years 2025 and 2026.")

    path_img = df_images["path"]
    date_img = df_images["date"]
    cloud_values = df_images["cloud"]

    date_excel = df_excel["Data"][1:].tolist()
    height_excel = df_excel["Cota(m)"][1:].tolist()

    # relate the quota with the date
    result_list = []
    for i in range(len(date_img)):
        if cloud_values[i] < 10.0:
            img_dt = to_dt(str(date_img[i]))
            best_j = -1
            min_diff = float('inf') 
            
            for j in range(len(date_excel)):
                excel_dt = to_dt(str(date_excel[j]))
                diff = abs((img_dt - excel_dt).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    best_j = j

            if best_j != -1 and min_diff <= (3 * 86400): 
                if date_img[i] not in ['2018-08-28', '2024-10-30', '2018-08-03', '2022-04-29']: # outliers
                    result_list.append([path_img[i], height_excel[best_j], cloud_values[i]])

    result_array = np.array(result_list, dtype=object)
    print(f"Successfully matched {len(result_array)} masks with historical water levels.")

    # compute accumulation map and extract border-pixel observations
    all_border_pixels = []
    accumulation = 0
    
    for path, quota, cloud in result_array:
        binary, border_pixels = process_mask(path)
        accumulation += binary

        if len(border_pixels) == 0:
            continue

        y = border_pixels[:, 0]
        x = border_pixels[:, 1]

        # Build final array directly
        pixels_with_quota = np.empty((len(border_pixels), 4), dtype=np.float64)
        pixels_with_quota[:, 0:2] = border_pixels
        pixels_with_quota[:, 2] = quota
        pixels_with_quota[:, 3] = accumulation[y, x]

        all_border_pixels.append(pixels_with_quota)

    all_border_pixels = np.concatenate(all_border_pixels, axis=0)

    # Extract observations from the accumulated border-pixel table
    # all_border_pixels columns: [y, x, quota (m), accumulation_count]
    border_ys      = all_border_pixels[:, 0].astype(int)
    border_xs      = all_border_pixels[:, 1].astype(int)
    border_heights = all_border_pixels[:, 2].astype(np.float32)

    # Grid dimensions from accumulation map
    H, W = accumulation.shape

    print(f"Grid size            : {H} x {W}")
    print(f"Total observations   : {len(border_ys):,}")
    print(f"Elevation range      : [{border_heights.min():.2f}, {border_heights.max():.2f}] m")

    # ── Run terrain reconstruction ────────────────────────────────────────
    DEM, loss_history = reconstruct_terrain(
        H=H, W=W,
        border_ys=border_ys,
        border_xs=border_xs,
        border_heights=border_heights,
        accumulation=accumulation,
        lambda_tv=5.0,
        n_iters=3000,
        lr=0.5,
        verbose=True,
        log_interval=500,
    )

    np.save("DEM", DEM)
    print("Reconstructed DEM saved to DEM.npy.")
    np.save("SAM", accumulation)
    print("Accumulation map saved to SAM.npy.")

    # ── Visualisation and Plot Saving ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Reconstructed DEM
    im0 = axes[0].imshow(DEM, cmap="terrain")
    axes[0].set_title("Reconstructed DEM (m)", fontsize=13)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], label="Elevation (m)")

    # Contour observations coloured by height
    sc = axes[1].scatter(
        border_xs, border_ys,
        c=border_heights, cmap="plasma",
        s=0.3, alpha=0.5,
    )
    axes[1].set_xlim(0, W)
    axes[1].set_ylim(H, 0)
    axes[1].set_title("Contour Observations\n(coloured by height)", fontsize=13)
    axes[1].set_aspect("equal")
    plt.colorbar(sc, ax=axes[1], label="Elevation (m)")

    # Water accumulation map
    im2 = axes[2].imshow(accumulation, cmap="hot")
    axes[2].set_title("Water Accumulation Map", fontsize=13)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], label="Count")

    plt.tight_layout()
    plt.savefig("reconstruction_results.png", dpi=150)
    print("Reconstruction visualisations saved to reconstruction_results.png.")
    plt.show()

    # ── Convergence curve ─────────────────────────────────────────────────
    iters      = [e["iter"]      for e in loss_history]
    total_loss = [e["loss"]      for e in loss_history]
    lc_vals    = [e["L_contour"] for e in loss_history]
    ltv_vals   = [e["L_tv"]      for e in loss_history]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(iters, total_loss, marker="o", label="Total loss")
    ax.plot(iters, lc_vals,    marker="s", label="L_contour")
    ax.plot(iters, ltv_vals,   marker="^", label="L_tv")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Optimisation Convergence")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("convergence_curve.png", dpi=150)
    print("Convergence curve saved to convergence_curve.png.")
    plt.show()