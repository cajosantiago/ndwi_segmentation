import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

def main():
    print("=" * 70)
    # Load files
    p_path = 'data/excel/predicted_elevation.csv'
    a_path = 'data/excel/cota_maranhao.csv'
    
    if not os.path.exists(p_path) or not os.path.exists(a_path):
        print("[ERROR] CSV files not found!")
        return

    df_p = pd.read_csv(p_path)
    df_a = pd.read_csv(a_path)

    # Convert to datetime
    df_p['date'] = pd.to_datetime(df_p['date'])
    df_a['date'] = pd.to_datetime(df_a['date'])

    # Filter to 2021-2022 range and exclude obvious database outliers (e.g. cota > 150m)
    start_date = '2021-01-01'
    end_date = '2022-12-31'
    df_p_filtered = df_p[(df_p['date'] >= start_date) & (df_p['date'] <= end_date)].copy()
    df_a_filtered = df_a[(df_a['date'] >= start_date) & (df_a['date'] <= end_date) & (df_a['cota'] <= 150)].copy()

    # Sort values chronologically
    df_p_filtered = df_p_filtered.sort_values(by='date').reset_index(drop=True)
    df_a_filtered = df_a_filtered.sort_values(by='date').reset_index(drop=True)

    # Apply a 3-point rolling median filter to remove transient outliers (e.g. single overpass segmentation errors)
    df_p_filtered['elevation_raw'] = df_p_filtered['elevation']
    df_p_filtered['elevation'] = df_p_filtered['elevation'].rolling(window=3, center=True, min_periods=1).median()

    # Preserve values at the edges (first and last elements remain unfiltered)
    if len(df_p_filtered) > 0:
        df_p_filtered.loc[df_p_filtered.index[0], 'elevation'] = df_p_filtered['elevation_raw'].iloc[0]
    if len(df_p_filtered) > 1:
        df_p_filtered.loc[df_p_filtered.index[-1], 'elevation'] = df_p_filtered['elevation_raw'].iloc[-1]

    print(f"Loaded {len(df_p_filtered)} predicted points and {len(df_a_filtered)} actual cota points.")

    # Match each predicted point with the closest actual cota point within 3 days
    matched_records = []
    for idx, row in df_p_filtered.iterrows():
        pred_date = row['date']
        pred_val = row['elevation']
        pred_raw = row['elevation_raw']
        
        # Find nearest date in actual cota
        time_diffs = (df_a_filtered['date'] - pred_date).abs()
        min_idx = time_diffs.idxmin()
        min_diff = time_diffs.min()
        
        if pd.notna(min_diff) and min_diff.days <= 3:
            actual_row = df_a_filtered.iloc[min_idx]
            matched_records.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'predicted_elevation_raw': pred_raw,
                'predicted_elevation_filtered': pred_val,
                'actual_cota': actual_row['cota'],
                'actual_cota_date': actual_row['date'].strftime('%Y-%m-%d'),
                'difference_days': min_diff.days
            })

    df_matched = pd.DataFrame(matched_records)
    
    # Save the merged/matched data to CSV
    out_csv = 'data/excel/elevation_comparison_2021_2022.csv'
    df_matched.to_csv(out_csv, index=False)
    print(f"Saved matched dataset to: {out_csv}")

    # Compute comparison metrics
    if not df_matched.empty:
        errors = df_matched['predicted_elevation_filtered'] - df_matched['actual_cota']
        mae = errors.abs().mean()
        rmse = np.sqrt((errors ** 2).mean())
        mean_bias = errors.mean()
        print("\n--- Match Statistics (Median-Filtered Predictions) ---")
        print(f"Total matched dates (within 3 days): {len(df_matched)}")
        print(f"Mean Bias (Pred - Actual)          : {mean_bias:+.3f} m")
        print(f"Mean Absolute Error (MAE)          : {mae:.3f} m")
        print(f"Root Mean Squared Error (RMSE)     : {rmse:.3f} m")
        print("-" * 50)
    else:
        print("\n[WARN] No overlapping dates matched within 3 days!")

    # ── Premium Visualisation ──
    print("Generating premium comparison plot...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)
    
    # Sleek dark mode / premium white grids
    ax.grid(True, linestyle="--", alpha=0.5, color="#ccc")
    
    # Plot Actual Elevation as a continuous smooth line
    ax.plot(
        df_a_filtered['date'], df_a_filtered['cota'],
        color="#2b6cb0", linewidth=3, alpha=0.85,
        label="Actual Elevation (cota_maranhao.csv)", zorder=1
    )
    
    # Plot RAW Predicted Elevation as faint semi-transparent scatter points
    ax.scatter(
        df_p_filtered['date'], df_p_filtered['elevation_raw'],
        color="#cbd5e0", edgecolors="#a0aec0", linewidths=1.0,
        s=30, alpha=0.4, label="Raw Predictions (with outliers)", zorder=2
    )

    # Plot FILTERED Predicted Elevation as scatter points and dashed line
    ax.plot(
        df_p_filtered['date'], df_p_filtered['elevation'],
        color="#dd6b20", linestyle="--", linewidth=1.5, alpha=0.8,
        zorder=3
    )
    ax.scatter(
        df_p_filtered['date'], df_p_filtered['elevation'],
        color="#ed8936", edgecolors="#c05621", linewidths=1.2,
        s=55, alpha=0.95, label="Median Filtered Predictions", zorder=4
    )

    # Annotate summary metrics on the plot inside a clean box
    if not df_matched.empty:
        textstr = '\n'.join((
            r'$\mathbf{Median-Filtered\ Stats\ (2021-2022)}$',
            f'Matched Observations: {len(df_matched)}',
            f'Mean Bias: {mean_bias:+.2f} m',
            f'MAE: {mae:.2f} m',
            f'RMSE: {rmse:.2f} m'
        ))
        props = dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='#e2e8f0', alpha=0.9)
        ax.text(0.03, 0.08, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props, color="#2d3748")

    # Title & Labels
    ax.set_title("Maranhão Reservoir Water Elevation: Actual vs. Predicted (2021 - 2022)", fontsize=16, fontweight="bold", pad=20, color="#1a202c")
    ax.set_xlabel("Date", fontsize=12, fontweight="semibold", labelpad=12, color="#2d3748")
    ax.set_ylabel("Elevation (meters)", fontsize=12, fontweight="semibold", labelpad=12, color="#2d3748")
    
    # Customize axis limits and format dates beautifully
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30, ha='right')
    
    ax.tick_params(axis='both', which='major', labelsize=10, colors="#4a5568")
    
    # Add a legend with premium styling
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=10.5, shadow=False)
    
    # Save the visualization plot
    out_plot = 'data/excel/elevation_comparison_2021_2022.png'
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    print(f"Successfully saved time series comparison plot to: {out_plot}")
    print("=" * 70)

if __name__ == "__main__":
    main()
