import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

# Add project root to path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))
os.environ["R_HOME"] = r"C:\\Users\\david\\Personal\\PROGRA~1\\R-44~1.2"

from modules.concentration_algorithms.by_papers import (  # noqa: E402
    multi_start_continuous_search)

# --------- Config ---------
DATA_FOLDER = "data"
REPORT_FOLDER = "tests\\results"
OUTPUT_CSV = os.path.join(REPORT_FOLDER, "test_metaheuristic_results.csv")
RADIUS_LIST = [200]
N_REPLICATES = 1
DISTANCE_TYPE = "euclidean"  # label for this algorithm
# --------------------------


def run_metaheuristic_benchmark() -> None:
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    datasets = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    results = []

    print("🚀 Starting multi_start_continuous_search benchmark...\n")

    for filename in datasets:
        file_path = os.path.join(DATA_FOLDER, filename)
        df = pd.read_csv(file_path)
        n = len(df)
        print(f"📄 Dataset: {filename} | {n} policies")

        for radius in RADIUS_LIST:
            # initialize metadata for this run
            result = {
                "dataset": filename,
                "n_policies": n,
                "radius_m": radius,
                "distance_type": DISTANCE_TYPE,
                "timestamp": datetime.now().isoformat(),
            }

            try:
                # run N_REPLICATES times and pick best
                best_sum = -np.inf
                best_group_size = 0
                best_center: Optional[Tuple[float, float]] = None
                start = time.perf_counter()

                print(
                    f"🔎 multi_start_continuous_search → "
                    f"{DISTANCE_TYPE.upper()} | Radius {radius}m"
                )
                for run in range(N_REPLICATES):
                    group, total_sum, center = multi_start_continuous_search(
                        df,
                        radius=radius,
                        delta0=200 * 2**7,
                        delta_min=50,
                        i_min=50_000,
                        random_state=run,
                    )
                    group_size = len(group)

                    # update if this replicate is better
                    if (total_sum > best_sum) or (
                        total_sum == best_sum and group_size > best_group_size
                    ):
                        best_sum = total_sum
                        best_group_size = group_size
                        best_center = center

                assert best_center is not None, "best_center never set"

                elapsed = time.perf_counter() - start

                # record results
                result.update(
                    {
                        "sum_serial": round(best_sum, 2),
                        "group_size_serial": best_group_size,
                        "time_serial_sec": round(elapsed, 4),
                        "lat_center": best_center[0],
                        "lon_center": best_center[1],
                    }
                )

            except Exception as e:
                print(f"   ❌ Metaheuristic failed: {e}")
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

    # save with the same columns and order as exhaustive
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
    run_metaheuristic_benchmark()
