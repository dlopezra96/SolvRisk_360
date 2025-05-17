import os
import sys
import pandas as pd
import time
from datetime import datetime
from typing import List, Literal

# allow imports from project root
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))

from modules.concentration_algorithms.new_methods import (  # noqa: E402
    fast_with_multilocal)

# --------- Config ---------
DATA_FOLDER = "data"
REPORT_FOLDER = "tests\\results"
OUTPUT_CSV = os.path.join(REPORT_FOLDER,
                          "test_fast_with_multilocal_results.csv")
RADIUS_LIST = [200]
DISTANCES: List[Literal["haversine", "euclidean"]] = ["haversine", "euclidean"]
# --------------------------


def run_benchmark() -> None:
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    datasets = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    results = []

    print("🚀 Starting fast_with_multilocal benchmark...\n")

    for filename in datasets:
        file_path = os.path.join(DATA_FOLDER, filename)
        df = pd.read_csv(file_path)
        n = len(df)
        print(f"📄 Dataset: {filename} | {n} policies")

        for radius in RADIUS_LIST:
            for metric in DISTANCES:
                # initialize record for this configuration
                result = {
                    "dataset": filename,
                    "n_policies": n,
                    "radius_m": radius,
                    "distance_type": metric,
                    "timestamp": datetime.now().isoformat(),
                }

                try:
                    print(
                        f"🔎 fast_with_multilocal → {metric.upper()} |"
                        f"Radius {radius}m"
                    )
                    start = time.perf_counter()

                    # run algorithm and unpack results
                    group, total_sum, center = fast_with_multilocal(
                        df,
                        radius=radius,
                        top_k=10,
                        delta0=200.0 * 2,
                        delta_min=25.0,
                        i_min=100,
                        metric=metric,
                    )

                    duration = time.perf_counter() - start
                    lat_center, lon_center = center

                    # record metrics
                    result.update(
                        {
                            "sum_serial": round(total_sum, 2),
                            "group_size_serial": len(group),
                            "time_serial_sec": round(duration, 4),
                            "lat_center": lat_center,
                            "lon_center": lon_center,
                        }
                    )

                except Exception as e:
                    print(f"   ❌ fast_with_multilocal failed: {e}")
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

    # save results with consistent column order
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
    pd.DataFrame(results, columns=cols).to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Done! Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_benchmark()
