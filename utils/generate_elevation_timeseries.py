
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

    df_p = pd.read_csv(p_path)

    # Convert to datetime
    df_p['date'] = pd.to_datetime(df_p['date'])

    # Sort values chronologically
    df_p = df_p.sort_values(by='date').reset_index(drop=True)

    # Apply a 3-point rolling median filter to remove transient outliers (e.g. single overpass segmentation errors)
    df_p['elevation'] = df_p['elevation'].rolling(window=3, center=True, min_periods=1).median()

    # Preserve values at the edges (first and last elements remain unfiltered)
    if len(df_p) > 0:
        df_p.loc[df_p.index[0], 'elevation'] = df_p['elevation'].iloc[0]
    if len(df_p) > 1:
        df_p.loc[df_p.index[-1], 'elevation'] = df_p['elevation'].iloc[-1]

    print(f"Loaded {len(df_p)} predicted points")
    
    # Save the merged/matched data to CSV
    out_csv = 'data/excel/predicted_elevation_filtered.csv'
    df_p.to_csv(out_csv, index=False)
    print(f"Saved matched dataset to: {out_csv}")

if __name__ == "__main__":
    main()
