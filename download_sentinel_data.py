#!/opt/conda/bin/python3.11
"""
download_sentinel_data.py

Automates download of Sentinel-2 L2A band data from Copernicus Data Space Ecosystem (CDSE) 
via SentinelHub API, based on a geographic bounding box and date ranges.

Usage:
    python download_sentinel_data.py --albufeira Montargil \
                                    --aoi-coords -8.1870 39.0389 -8.0549 39.1771 \
                                    --start-date 2023-04-01 \
                                    --end-date 2023-04-15 \
                                    --max-cloud-cover 30
"""

import os
import sys
import argparse
import json
import getpass
import numpy as np
import tifffile
from datetime import datetime, timedelta
from collections import defaultdict

from sentinelhub import (
    SHConfig,
    DataCollection,
    SentinelHubCatalog,
    SentinelHubRequest,
    BBox,
    bbox_to_dimensions,
    CRS,
    MimeType,
)

# EVALSCRIPT — returns raw float bands in standard orders
EVALSCRIPT_RAW = """
//VERSION=3
function setup() {
  return {
    input:  [{ bands: ["B02","B03","B04","B05","B08","SCL","dataMask"] }],
    output: [
      { id: "raw_bands", bands: 6, sampleType: "FLOAT32" },
      { id: "dataMask",  bands: 1, sampleType: "UINT8"   }
    ]
  };
}
function evaluatePixel(s) {
  // Order of bands in output TIFF:
  // 0:B02, 1:B03, 2:B04, 3:B05, 4:B08, 5:SCL
  return {
    raw_bands: [s.B02, s.B03, s.B04, s.B05, s.B08, s.SCL],
    dataMask:  [s.dataMask]
  };
}
"""

def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 L2A band data from SentinelHub (CDSE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--albufeira",
        required=True,
        help="Name of the reservoir/dam (used to name the output directory)."
    )
    parser.add_argument(
        "--aoi-coords",
        nargs=4,
        type=float,
        required=True,
        help="4 WGS84 coordinates defining the BBox: min_lon min_lat max_lon max_lat"
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--max-cloud-cover",
        type=float,
        default=90.0,
        help="Maximum cloud cover percentage allowed for search results."
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=10.0,
        help="Pixel resolution in meters."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output directory. Defaults to data/sentinelhub/Bandas/{albufeira}."
    )
    parser.add_argument(
        "--client-id",
        default=None,
        help="SentinelHub Client ID. If not provided, reads SH_CLIENT_ID env var or loaded profiles."
    )
    parser.add_argument(
        "--client-secret",
        default=None,
        help="SentinelHub Client Secret. If not provided, reads SH_CLIENT_SECRET env var or loaded profiles."
    )

    args = parser.parse_args()

    # 1. Setup SentinelHub Configuration
    config = SHConfig()
    
    client_id = args.client_id or os.environ.get("SH_CLIENT_ID")
    client_secret = args.client_secret or os.environ.get("SH_CLIENT_SECRET")

    # If credentials not found in CLI args or env vars, try loading the cdse profile settings
    if not client_id or not client_secret:
        try:
            profile_config = SHConfig(profile="cdse")
            client_id = client_id or profile_config.sh_client_id
            client_secret = client_secret or profile_config.sh_client_secret
        except Exception:
            pass

    # Prompt interactively if still missing and running in interactive terminal
    if not client_id or not client_secret:
        if sys.stdin.isatty():
            print("SentinelHub API credentials not found in CLI arguments, environment variables, or profiles.")
            client_id = input("SentinelHub client id: ").strip()
            client_secret = getpass.getpass("SentinelHub client secret: ").strip()
        else:
            print("[ERROR] SentinelHub API credentials are required. Please provide them via "
                  "--client-id/--client-secret CLI parameters or set SH_CLIENT_ID/SH_CLIENT_SECRET environment variables.", 
                  file=sys.stderr)
            sys.exit(1)

    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.save("cdse")  # Save to profile for persistence

    # 2. Output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"/home/csantiago/data/sentinelhub/Bandas/{args.albufeira}"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("  SENTINEL-2 L2A DOWNLOAD PIPELINE (CDSE)")
    print("="*80)
    print(f"  Albufeira       : {args.albufeira}")
    print(f"  AOI Bounding Box: {args.aoi_coords}")
    print(f"  Start Date      : {args.start_date}")
    print(f"  End Date        : {args.end_date}")
    print(f"  Max Cloud Cover : {args.max_cloud_cover}%")
    print(f"  Resolution      : {args.resolution}m")
    print(f"  Output Directory: {output_dir}")
    print("="*80 + "\n")

    # 3. Setup AOI Geometry
    aoi_bbox = BBox(bbox=args.aoi_coords, crs=CRS.WGS84)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution=args.resolution)
    print(f"Grid Dimensions : {aoi_size} px")

    # 4. Query Copernicus Catalog
    print("Querying Copernicus catalog...")
    catalog = SentinelHubCatalog(config=config)
    try:
        results = list(catalog.search(
            DataCollection.SENTINEL2_L2A,
            bbox=aoi_bbox,
            time=(args.start_date, args.end_date),
            filter=f"eo:cloud_cover < {args.max_cloud_cover}",
            filter_lang="cql2-text",
            fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []},
        ))
    except Exception as e:
        print(f"[ERROR] Catalog query failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(results)} total candidate scenes.")

    # 5. Group by day and choose the lowest cloud cover scene
    daily_groups = defaultdict(list)
    for item in results:
        dt = datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00"))
        daily_groups[dt.date()].append(item)

    chosen = [min(items, key=lambda it: it["properties"].get("eo:cloud_cover", 1e9))
              for items in daily_groups.values()]
    chosen.sort(key=lambda it: it["properties"]["datetime"])
    print(f"Selected {len(chosen)} unique daily scenes for download.\n")

    if not chosen:
        print("No matching scenes found for the specified parameters.")
        sys.exit(0)

    # 6. Loop and download scenes
    buffer = timedelta(minutes=30)
    index = []
    BANDS_REQUESTED = ["B02", "B03", "B04", "B05", "B08", "SCL"]
    BAND_IDX = {"B02": 0, "B03": 1, "B04": 2, "B05": 3, "B08": 4, "SCL": 5}

    for idx, item in enumerate(chosen, start=1):
        acq_iso     = item["properties"]["datetime"]
        acq_dt      = datetime.fromisoformat(acq_iso.replace("Z", "+00:00"))
        cloud_cover = item["properties"].get("eo:cloud_cover", None)
        date_str    = acq_dt.strftime("%Y-%m-%d")
        ts_str      = acq_dt.strftime("%Y%m%dT%H%M%SZ")
        cc_tag      = f"cc{cloud_cover:.1f}pct" if cloud_cover is not None else "ccNA"

        tif_name = f"{ts_str}_{cc_tag}_bands.tif"
        tif_path = os.path.join(output_dir, tif_name)

        # Skip download if the file already exists
        if os.path.exists(tif_path):
            print(f"[{idx:03d}/{len(chosen)}] {date_str} already exists — skipping.")
            index.append({"date": date_str, "ts": ts_str, "cloud_cover": cloud_cover, "tif": tif_name})
            continue

        t_from = (acq_dt - buffer).strftime("%Y-%m-%dT%H:%M:%SZ")
        t_to   = (acq_dt + buffer).strftime("%Y-%m-%dT%H:%M:%SZ")

        print(f"[{idx:03d}/{len(chosen)}] {date_str} (cloud={cloud_cover:.1f}%) ... ", end="")
        sys.stdout.flush()

        try:
            request = SentinelHubRequest(
                evalscript=EVALSCRIPT_RAW,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A.define_from(
                            name="s2", service_url="https://sh.dataspace.copernicus.eu"
                        ),
                        time_interval=(t_from, t_to),
                        other_args={"dataFilter": {"mosaickingOrder": "mostRecent"}},
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("raw_bands", MimeType.TIFF),
                    SentinelHubRequest.output_response("dataMask",  MimeType.TIFF),
                ],
                bbox=aoi_bbox,
                size=aoi_size,
                config=config,
            )
            data = request.get_data()[0]
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        raw = data["raw_bands.tif"].astype(np.float32)  # shape (H, W, 6)

        # Save TIFF transposed to standard geospatial channel order (bands, H, W)
        tifffile.imwrite(tif_path, raw.transpose(2, 0, 1))

        index.append({"date": date_str, "ts": ts_str, "cloud_cover": cloud_cover, "tif": tif_name})
        print(f"OK → {tif_name}")

    # 7. Write index.json metadata catalog
    index_path = os.path.join(output_dir, "index.json")
    try:
        with open(index_path, "w") as f:
            json.dump({
                "albufeira":       args.albufeira,
                "bands":           BANDS_REQUESTED,
                "band_idx":        BAND_IDX,
                "start_date":      args.start_date,
                "end_date":        args.end_date,
                "resolution_m":    args.resolution,
                "aoi_coords":      args.aoi_coords,
                "aoi_size_px":     list(aoi_size),
                "scenes":          index,
            }, f, indent=2)
        print(f"\nMetadata index saved: {index_path}")
    except Exception as e:
        print(f"\n[WARNING] Failed to write index metadata file: {e}")

    print("\n" + "="*80)
    print(f"  DOWNLOAD PROCESS COMPLETED SUCCESSFULLY")
    print(f"  Saved {len(index)} total scenes to {output_dir}/")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
