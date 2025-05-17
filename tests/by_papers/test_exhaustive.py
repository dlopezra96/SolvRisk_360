import os
import sys
import pandas as pd
import time
from datetime import datetime
from typing import List, Literal


# Add project root to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

os.environ["R_HOME"] = r"C:\\Users\\david\\Personal\\PROGRA~1\\R-44~1.2"

from modules.concentration_algorithms.by_papers import (  # noqa: E402
    exhaustive_concentration)

# --------- Config ---------
DATA_FOLDER = "data"
REPORT_FOLDER = "tests\\results"
OUTPUT_CSV = os.path.join(REPORT_FOLDER, "test_exhaustive_results.csv")
RADIUS_LIST = [200]
DISTANCES: List[Literal["haversine", "euclidean"]] = ["haversine", "euclidean"]
# --------------------------


def run_benchmark() -> None:
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    datasets = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    results = []

    print("🚀 Starting exhaustive_concentration benchmark...\n")

    for filename in datasets:
        file_path = os.path.join(DATA_FOLDER, filename)
        df = pd.read_csv(file_path)
        print(f"📄 Dataset: {filename} | {len(df)} policies")

        for radius in RADIUS_LIST:
            for distance_type in DISTANCES:
                result = {
                    "dataset": filename,
                    "n_policies": len(df),
                    "radius_m": radius,
                    "distance_type": distance_type,
                    "timestamp": datetime.now().isoformat(),
                }

                # --- Serial only ---
                try:
                    print(f"🔎 Serial → {distance_type.upper()} |"
                          f"Radius {radius}m")
                    start_time = time.perf_counter()
                    group, total_sum, best_center = exhaustive_concentration(
                        df, distance_type=distance_type, radius=radius
                    )
                    duration = time.perf_counter() - start_time

                    result.update(
                        {
                            "sum_serial": round(total_sum, 2),
                            "group_size_serial": len(group),
                            "time_serial_sec": round(duration, 4),
                            "lat_center": best_center[0],
                            "lon_center": best_center[1],
                        }
                    )

                except Exception as e:
                    print(f"   ❌ Serial failed: {str(e)}")
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

    # Save results to CSV
    print("\n💾 Saving CSV to:", OUTPUT_CSV)
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Done! Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_benchmark()
