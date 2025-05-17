"""concentration_algorithms/by_papers.py."""

import numpy as np
import pandas as pd
import time
from typing import Tuple, Optional
from numpy.typing import NDArray
from modules.distance_utils import haversine_matrix, euclidean_matrix
from modules.distance_utils import R

# rpy2 imports for the grid‐search only
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter

# Load R packages just once at module import
importr("readr")
spatialrisk = importr("spatialrisk")

"""
Collection of classic concentration-risk algorithms:
 1) Exhaustive brute-force (Badal-Valero E. et al.)
 2) Multi-start local search (Gomes et al.)
 3) Grid search via R0s spatialrisk package (Martin Haringa)
"""


# ────────────────────────────────────────────────────────────────────────────────
# 1) Exhaustive brute‑force algorithm
# ────────────────────────────────────────────────────────────────────────────────
def exhaustive_concentration(
    df: pd.DataFrame, distance_type: str = "haversine", radius: float = 200
) -> Tuple[NDArray[np.int_], float, Tuple[float, float]]:
    """Exhaustive brute-force algorithm.

    Brute-force algorithm to find the group of policies within a given radius
    that maximizes the sum of insured values, using either Haversine
    or Euclidean distance.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'lat', 'lon', and 'insured_sum'.
    - distance_type (str): Type of distance to use ("haversine" or
    "euclidean").
    - radius (float): Maximum distance in meters to consider policies
    as a group.

    Returns:
    - best_group (np.ndarray): Indices of the higuest-sum cluster.
    - max_sum (float): Maximum total insured sum of that cluster.
    - best_center (Tuple[float, float]): Lat and lon where is located
    the center of highest cluster.
    """
    # Compute the distance matrix
    if distance_type == "haversine":
        dist_matrix = haversine_matrix(df["lat"].values, df["lon"].values)
    elif distance_type == "euclidean":
        dist_matrix = euclidean_matrix(df["lat"].values, df["lon"].values)
    else:
        raise ValueError(
            "Unsupported distance_type. Use 'haversine' or "
            "'euclidean' instead."
        )

    max_sum = 0
    best_group: NDArray[np.int_] = np.array([], dtype=int)
    best_center = (0.0, 0.0)

    for i in range(len(df)):
        group_idx = np.where(dist_matrix[i] <= radius)[0]
        total_sum = df.iloc[group_idx]["insured_sum"].sum()

        if total_sum > max_sum:
            max_sum = total_sum
            best_group = group_idx
            best_center = (float(df.iloc[i]["lat"]), float(df.iloc[i]["lon"]))

    return best_group, float(max_sum), best_center


# ────────────────────────────────────────────────────────────────────────────────
# 2) Local‐search metaheuristic
# ────────────────────────────────────────────────────────────────────────────────
def multi_start_continuous_search(
    df: pd.DataFrame,
    radius: float = 200.0,
    delta0: float = 200.0 * 2**7,
    delta_min: float = 25.0,
    i_min: int = 50_000,
    random_state: Optional[int] = None,
) -> Tuple[NDArray[np.int_], float, Tuple[float, float]]:
    """
    Replication of Fire Risk Meta-heuristic (Gomes et al. 2018) – Algorithm 1.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'lat', 'lon', 'insured_sum'.
    - radius (float): Circle radius in meters.
    - delta0 (float): Initial pattern-search step size Δ₀ in meters.
    - delta_min (float): Minimum step size Δₘᵢₙ to stop local search.
    - i_min (int): Minimum total pattern-search iterations globally.
    - random_state (int): Seed for reproducibility.

    Returns:
    - best_group (np.ndarray): Indices of the highest-sum cluster.
    - best_sum (float): Maximum total insured sum of that cluster.
    - best_center (Tuple[float, float]): Lat and lon of the continuous
    optimal center.
    """
    rng = np.random.default_rng(random_state)
    insured = df["insured_sum"].values

    # 1: Determine maximum single-point sum
    best = insured.max()
    # 2: Initial step size
    delta_0 = delta0

    # Forward equirectangular projection of lat/lon to x,y coordinates
    lat_rad = np.deg2rad(df["lat"].values)
    lon_rad = np.deg2rad(df["lon"].values)
    lat0 = lat_rad.mean()
    lon0 = lon_rad.mean()
    xs = R * (lon_rad - lon0) * np.cos(lat0)
    ys = R * (lat_rad - lat0)
    radius2 = radius**2

    # Define objective f(s): sum insured within radius of s
    def f(s: Tuple[float, float]) -> float:
        dx = xs - s[0]
        dy = ys - s[1]
        total: float = float(insured[(dx * dx + dy * dy) <= radius2].sum())
        return total

    # 3: Generate initial center s0
    x0 = rng.uniform(xs.min(), xs.max())
    y0 = rng.uniform(ys.min(), ys.max())
    s0 = (x0, y0)

    # 4: Initialize best values
    best_center_xy = s0
    best_sum = f(s0)
    best_group = np.where((xs - x0) ** 2 + (ys - y0) ** 2 <= radius2)[0]

    i = 0  # global iteration counter
    # 6: Main loop: until enough iterations and sum reaches Best
    while (i < i_min) or (best_sum < best):
        if i > 0:
            si = (rng.uniform(xs.min(), xs.max()),
                  rng.uniform(ys.min(), ys.max()))
            delta = delta_0
        else:
            si = s0
            delta = delta_0

        consec_success = 0
        # 11: Local search until delta below minimum
        while delta >= delta_min:
            current_sum = f(si)
            best_n, best_n_sum = si, current_sum
            x_si, y_si = si
            # 12: Generate neighbors N(si)
            neighbors = [
                (x_si + delta, y_si),
                (x_si - delta, y_si),
                (x_si, y_si + delta),
                (x_si, y_si - delta),
            ]
            # 13: Pick best neighbor
            for nb in neighbors:
                s_nb_sum = f(nb)
                if s_nb_sum > best_n_sum:
                    best_n, best_n_sum = nb, s_nb_sum

            # 14–20: Success: move (& maybe double delta)
            if best_n_sum > current_sum:
                si = best_n
                consec_success += 1
                if consec_success >= 2:
                    delta *= 2
                    consec_success = 0
            else:
                # 21–24: Failure: shrink delta
                delta /= 2
                consec_success = 0

            i += 1  # 25: increment global iteration

        # 27: Update global best if improved
        final_sum = f(si)
        if final_sum > best_sum:
            best_sum = final_sum
            best_center_xy = si
            best_group = np.where((xs - si[0]) ** 2
                                  + (ys - si[1]) ** 2 <= radius2)[0]

    # 29: Inverse projection: convert best_center_xy back to lat/lon
    x_c, y_c = best_center_xy
    lat_center = np.rad2deg(y_c / R + lat0)
    lon_center = np.rad2deg(x_c / (R * np.cos(lat0)) + lon0)
    best_center = (lat_center, lon_center)

    return best_group, float(best_sum), best_center


# ────────────────────────────────────────────────────────────────────────────────
# 3) Grid search algorithm
# ────────────────────────────────────────────────────────────────────────────────
def grid_search_r(
    csv_path: str, radius: float = 200.0, grid_distance: float = 25
) -> Tuple[int, float, Tuple[float, float], float]:
    """
    Fast grid search via R’s spatialrisk::highest_concentration().

    Parameters:
    - csv_path (str): Path to the CSV file (must contain columns lat,
    lon, insured_sum).
    - radius (float): Circle radius in meters (default: 200).
    - grid_distance (float): Grid‐step in meters (default: 25).

    Returns:
    - best_group (int): Number of policies within that circle.
    - best_sum (float): Maximum total insured sum of that cluster.
    - best_center (Tuple[float, float]): Lat and lon of the continuous
    optimal center.
    - elapsed_time (float): Time spent in seconds running the C++ code.
    """
    # 1) Normalize the file path for R (forward slashes)
    r_csv = csv_path.replace("\\", "/")

    # 2) Read the CSV in R with readr::read_csv, suppressing messages
    read_cmd = (
        "suppressMessages("
        f"  readr::read_csv('{r_csv}', show_col_types=FALSE, progress=FALSE)"
        ")"
    )
    with localconverter(default_converter):
        r_df = r(read_cmd)

        # 3) Create a symbol for the insured_sum column
        insured_sum = r("as.symbol")("insured_sum")

        # 4) Call highest_concentration() in the C++ backend and time it
        t0 = time.time()
        hconc = spatialrisk.highest_concentration(
            r_df,
            insured_sum,
            radius=radius,
            grid_distance=grid_distance,
            display_progress=False,
        )
        elapsed_time = time.time() - t0

        # 5) Extract the optimal center coordinates from the returned tibble
        lon = float(hconc.rx2("lon")[0])
        lat = float(hconc.rx2("lat")[0])
        best_center = (lat, lon)

        # 6) Retrieve the actual policies within that circle
        pts = spatialrisk.points_in_circle(
            r_df, lon_center=lon, lat_center=lat, radius=radius
        )

        # 7) Compute sum and count inside R (no pandas conversions)
        best_sum = float(r["sum"](pts.rx2("insured_sum"))[0])
        best_group = int(r["nrow"](pts)[0])

    return best_group, best_sum, best_center, elapsed_time
