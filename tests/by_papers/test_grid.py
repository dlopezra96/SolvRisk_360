import os
import sys
import pandas as pd
from datetime import datetime

# Allow imports from project root
tabspath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(tabspath)
# Ensure R_HOME is set
os.environ["R_HOME"] = r"C:\\Users\\david\\Personal\\PROGRA~1\\R-44~1.2"

from modules.concentration_algorithms.by_papers import (  # noqa: E402
    grid_search_r)


# --------- Config ---------
DATA_FOLDER = "data"
REPORT_FOLDER = "tests\\results"
OUTPUT_CSV = os.path.join(REPORT_FOLDER, "test_grid_results.csv")
RADIUS_LIST = [200]
DISTANCE_TYPE = "haversine"  # spatialrisk only does haversine
GRID_DISTANCE = 25
# --------------------------


def run_grid_benchmark() -> None:
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    datasets = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    results = []

    print("🚀 Starting grid_search_r (spatialrisk) benchmark…\n")

    for filename in datasets:
        file_path = os.path.join(DATA_FOLDER, filename)
        df = pd.read_csv(file_path)  # for count
        print(f"📄 Dataset: {filename} | {len(df)} policies")

        for radius in RADIUS_LIST:
            result = {
                "dataset": filename,
                "n_policies": len(df),
                "radius_m": radius,
                "distance_type": DISTANCE_TYPE,
                "timestamp": datetime.now().isoformat(),
            }

            try:
                print(f"🔎 Grid → {DISTANCE_TYPE.upper()} | Radius {radius}m")
                # Unpack 4 values
                group_size, total_sum, center, duration = grid_search_r(
                    file_path, radius=radius, grid_distance=GRID_DISTANCE
                )
                lat_center, lon_center = center

                result.update(
                    {
                        "sum_serial": round(total_sum, 2),
                        "group_size_serial": group_size,
                        "time_serial_sec": round(duration, 4),
                        "lat_center": lat_center,
                        "lon_center": lon_center,
                    }
                )
            except Exception as e:
                print(f"   ❌ Grid failed: {e}")
                result.update(
                    {
                        "sum_serial": "ERROR",
                        "group_size_serial": "ERROR",
                        "time_serial_sec": "ERROR",
                        "lat_center": "ERROR",
                        "lon_center": "ERROR",
                    }
                )

            results.append(result)

    # Save CSV
    df_out = pd.DataFrame(results)
    # Ensure columns order
    cols = [
        "dataset",
        "n_policies",
        "radius_m",
        "distance_type",
        "timestamp",
        "sum_serial",
        "group_size_serial",
        "time_serial_sec",
        "lat_center",
        "lon_center",
    ]
    df_out[cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Done! Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_grid_benchmark()
