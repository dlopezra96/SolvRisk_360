"""New algorithms: exhaustive, grid, and multilocal."""
from typing import Tuple, Literal
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from modules.distance_utils import R
from modules.spatial_index import (
    haversine_coords,
    euclidean_coords,
    invert_euclidean,
    build_balltree,
    tree_radius,
    create_grid_around_point,
    create_incidence_matrix,
)


# ────────────────────────────────────────────────────────────────────────────────
# 1) Fast Exhaustive brute‑force algorithm
# ────────────────────────────────────────────────────────────────────────────────
def fast_exhaustive(
    df: pd.DataFrame,
    radius_m: float = 200.0,
    metric: Literal["haversine", "euclidean"] = "haversine",
) -> Tuple[NDArray[np.int_], float, Tuple[float, float]]:
    """
    Fast exhaustive search for highest insured-sum cluster using a BallTree.

    Steps:
    1) Build a BallTree from the DataFrame coordinates.
    2) Query radius for ALL points at once to get neighbor lists.
    3) Construct a sparse incidence matrix and multiply by values vector.
    4) Identify the center with maximum sum, return its indices, sum, and
    center coords.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'lat', 'lon', and
    'insured_sum' columns.
    - radius_m (float): Search radius in meters.
    - metric (str): Distance metric to use ("haversine" or "euclidean").

    Returns:
    - best_group (np.ndarray): Indices of policies within the optimal circle.
    - best_sum (float): Maximum total insured sum of that cluster.
    - best_center (Tuple[float, float]): (lat, lon) of the center achieving
    best_sum.
    """
    # Extract coordinate and value arrays
    lat = df["lat"].values
    lon = df["lon"].values
    vals = df["insured_sum"].values
    n = len(df)

    # Choose projection and metric
    if metric == "haversine":
        coords = haversine_coords(lat, lon)
    else:
        coords = euclidean_coords(lat, lon)

    # Build BallTree and radius value
    tree = build_balltree(coords, metric=metric)
    r = tree_radius(radius_m, metric)

    # Query neighbors for all points
    neighbors_list = tree.query_radius(coords, r=r, return_distance=False)

    # Build a sparse incidence matrix in COO format
    rows, cols = [], []
    for i, nbr in enumerate(neighbors_list):
        rows.extend([i] * len(nbr))
        cols.extend(nbr.tolist())
    data = np.ones(len(rows), dtype=int)
    matrix = coo_matrix((data, (rows, cols)), shape=(n, n))

    # Compute sum of insured values for each potential center
    sums = matrix.dot(vals)

    # Find the index of the best center
    best_idx = int(np.argmax(sums))
    best_sum = float(sums[best_idx])
    best_group = neighbors_list[best_idx]

    # Retrieve lat/lon of that best center
    best_center = (float(df.iloc[best_idx]["lat"]),
                   float(df.iloc[best_idx]["lon"]))

    return best_group, best_sum, best_center


# ────────────────────────────────────────────────────────────────────────────────
# 2) Fast exhaustive with grid search
# ────────────────────────────────────────────────────────────────────────────────
def fast_with_grid(
    df: pd.DataFrame,
    radius: float = 200.0,
    grid_distance: float = 25.0,
    top_k: int = 10,
    metric: Literal["haversine", "euclidean"] = "haversine",
) -> Tuple[NDArray[np.int_], float, Tuple[float, float]]:
    """Fast exhaustive with grid search algorithm.

    Hybrid fast exhaustive + grid-refinement search for highest
    insured-sum cluster.

    Steps:
    1) Run a fast_exhaustive-style pass to score every original point,
       then pick the top_k “anchor” indices by insured_sum.
    2) For each anchor, generate a fine-grained local grid of candidate centers
       in a square of side ±radius around the anchor (step=grid_step).
    3) Query the BallTree at all grid points, build an incidence matrix, and
       compute insured-sum at each grid center.
    4) Compare all continuous sums and pick the single best center across
    all anchors.

    Parameters:
    - df (pd.DataFrame): must contain 'lat', 'lon', and 'insured_sum' columns.
    - radius_m (float): radius of each search circle in meters.
    - grid_step (float): spacing between grid points in meters.
    - top_k (int): number of top discrete anchors to refine.
    - metric (str): “haversine” or “euclidean” distance metric.

    Returns:
    - best_group (np.ndarray): Indices of policies within the best circle.
    - best_sum (float): Total insured_sum of that best circle.
    - best_center (Tuple[float, float]): (lat, lon) of the continuous
    optimal center.
    """
    # Extract raw array
    lat = df["lat"].values
    lon = df["lon"].values

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0 = lat_rad.mean()
    lon0 = lon_rad.mean()

    vals = df["insured_sum"].values
    n = len(df)

    # 1) Build BallTree and score every original point
    if metric == "haversine":
        coords_tree = haversine_coords(lat, lon)
    else:
        coords_tree = euclidean_coords(lat, lon)

    tree = build_balltree(coords_tree, metric=metric)
    r = tree_radius(radius, metric)

    # Query neighbors for all points at once
    nbrs_list = tree.query_radius(coords_tree, r=r, return_distance=False)

    # Build sparse incidence matrix of shape (n x n)
    rows, cols = [], []
    for i, nbrs in enumerate(nbrs_list):
        rows.extend([i] * len(nbrs))
        cols.extend(nbrs.tolist())
    matrix = coo_matrix((np.ones(len(rows), dtype=int), (rows, cols)),
                        shape=(n, n))

    # Discrete sums at each original point
    disc_sums = matrix.dot(vals)

    # Select top_k anchors by discrete sum
    top_idxs = np.argpartition(-disc_sums, top_k - 1)[:top_k]
    top_idxs = top_idxs[np.argsort(-disc_sums[top_idxs])]

    # Record the best discrete result as a fallback
    best_disc_idx = int(top_idxs[0])
    best_disc_sum = float(disc_sums[best_disc_idx])
    best_disc_group = nbrs_list[best_disc_idx]

    # Prepare to track the best continous result
    best_cont_sum = best_disc_sum
    best_cont_group = best_disc_group
    best_center = (
        float(df.iloc[best_disc_idx]["lat"]),
        float(df.iloc[best_disc_idx]["lon"]),
    )

    # 2) & 3) Refine each anchor via local grid
    for anchor_idx in top_idxs:
        # Create a grid of candidate centers around the anchor
        anchor_pt = coords_tree[anchor_idx]
        grid = create_grid_around_point(anchor_pt, r, grid_distance, metric)

        # Query neighbors for all grid points
        grid_nbrs = tree.query_radius(grid, r=r, return_distance=False)

        # Build incidence matrix for this grid
        incidence_matrix = create_incidence_matrix(grid_nbrs, n, len(grid))

        # Compute continuous sums
        grid_sums = incidence_matrix.dot(vals)

        # Find best grid point for this anchor
        local_best_idx = int(np.argmax(grid_sums))
        local_best_sum = float(grid_sums[local_best_idx])

        # If this local best beats the global best, update
        if local_best_sum > best_cont_sum:
            best_cont_sum = local_best_sum
            best_cont_group = grid_nbrs[local_best_idx]

            if metric == "haversine":
                # Convert that grid point back to lat/lon
                best_center = (
                    float(np.degrees(grid[local_best_idx][0])),
                    float(np.degrees(grid[local_best_idx][1])),
                )
            else:
                best_center = invert_euclidean(tuple(grid[local_best_idx]),
                                               lat0, lon0)

    # 4) Return whichever was best
    return best_cont_group, best_cont_sum, best_center


# ────────────────────────────────────────────────────────────────────────────────
# 3) Fast exhaustive with multilocal search
# ────────────────────────────────────────────────────────────────────────────────
def fast_with_multilocal(
    df: pd.DataFrame,
    radius: float = 200.0,
    top_k: int = 10,
    # micro‐multistart params:
    delta0: float = 200.0 * 2,  # initial step size (smaller)
    delta_min: float = 25.0,  # minimum step size to stop
    i_min: int = 100,  # max pattern-search iterations
    metric: Literal["haversine", "euclidean"] = "haversine",
) -> Tuple[NDArray[np.int_], float, Tuple[float, float]]:
    """
    Fast exhaustive search combined with micro multi-start pattern search.

    Steps:
    1) Compute discrete sums for every point (BallTree + sparse dot).
    2) Select the top_k anchors by discrete sum.
    3) Around each anchor, run a short pattern-search:
       - start at the anchor's XY coordinate
       - move in cardinal directions adapting step size
       - perform up to i_min iterations
       - track the best continuous sum and center
    4) Return the best result across all anchors.

    Parameters:
    - df (pd.DataFrame): must contain 'lat', 'lon', 'insured_sum'.
    - radius (float): search circle radius in meters.
    - top_k (int): number of top discrete anchors to refine.
    - delta0 (float): initial step size for local search.
    - delta_min (float): minimum step size to stop local search.
    - i_min (int): max number of pattern-search iterations per anchor.
    - metric (str): 'haversine' or 'euclidean' distance metric.

    Returns:
    - best_group (np.ndarray): Indices of policies in the best circle.
    - best_sum (float): Total insured_sum within that circle.
    - best_center (Tuple[float, float]): (lat, lon) of the continuous
    optimal center.
    """
    # 0) Extract arrays
    lat = df["lat"].values
    lon = df["lon"].values
    vals = df["insured_sum"].values
    n = len(df)

    # Precompute radians always
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0 = lat_rad.mean()
    lon0 = lon_rad.mean()

    # 1) Discrete BallTree + sparse dot
    if metric == "haversine":
        coords_tree = haversine_coords(lat, lon)
    else:
        coords_tree = euclidean_coords(lat, lon)
    tree = build_balltree(coords_tree, metric=metric)
    r_tree = tree_radius(radius, metric)

    nbrs_list = tree.query_radius(coords_tree, r=r_tree, return_distance=False)
    rows, cols = [], []
    for i, nbrs in enumerate(nbrs_list):
        rows.extend([i] * len(nbrs))
        cols.extend(nbrs.tolist())
    matrix = coo_matrix((np.ones(len(rows), int), (rows, cols)),
                        shape=(n, n))
    disc_sums = matrix.dot(vals)

    # Select top_k discrete anchors
    top_idxs = np.argpartition(-disc_sums, top_k - 1)[:top_k]
    top_idxs = top_idxs[np.argsort(-disc_sums[top_idxs])]

    # Record best discrete fallback
    best_disc_idx = int(top_idxs[0])
    best_disc_sum = float(disc_sums[best_disc_idx])
    best_disc_group = nbrs_list[best_disc_idx]

    # 2) & 3) Pure haversine or euclidean continuous setup
    if metric == "haversine":
        # → Use haversine distance in radians
        coords_cont = coords_tree
        thr = radius / R

        def f(pt: Tuple[float, float]) -> float:
            lat_c, lon_c = pt
            dlat = lat_rad - lat_c
            dlon = lon_rad - lon_c
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat_rad) * np.cos(lat_c) * np.sin(dlon / 2) ** 2
            )
            d = 2 * np.arcsin(np.sqrt(a))
            total: float = float(vals[d <= thr].sum())
            return total

        delta0_cont = delta0 / R
        delta_min_cont = delta_min / R
    else:
        # → Use euclidean distance in meters
        coords_cont = euclidean_coords(lat, lon)
        xs, ys = coords_cont[:, 0], coords_cont[:, 1]
        thr = radius**2

        def f(pt: Tuple[float, float]) -> float:
            dx = xs - pt[0]
            dy = ys - pt[1]
            total: float = float(vals[(dx * dx + dy * dy) <= thr].sum())
            return total

        delta0_cont = delta0
        delta_min_cont = delta_min

    # Initialize best continuous
    best_cont_sum = best_disc_sum
    best_cont_group = best_disc_group

    if metric == "haversine":
        best_center = (
            float(np.degrees(coords_cont[best_disc_idx][0])),
            float(np.degrees(coords_cont[best_disc_idx][1])),
        )
    else:
        best_center = invert_euclidean(tuple(coords_cont[best_disc_idx]),
                                       lat0, lon0)

    # 4) Micro multi‐start pattern search
    for anchor in top_idxs:
        s = tuple(coords_cont[anchor])
        current_sum = f(s)
        delta = delta0_cont
        consec = 0
        iters = 0

        while iters < i_min and delta >= delta_min_cont:
            iters += 1
            lat0_s, lon0_s = s
            # Generate neigh in 4 directions
            if metric == "haversine":
                # Steps in radians
                neighbors = [
                    (lat0_s + delta, lon0_s),
                    (lat0_s - delta, lon0_s),
                    (lat0_s, lon0_s + delta),
                    (lat0_s, lon0_s - delta),
                ]
            else:
                # Steps in meters
                neighbors = [
                    (lat0_s + delta, lon0_s),
                    (lat0_s - delta, lon0_s),
                    (lat0_s, lon0_s + delta),
                    (lat0_s, lon0_s - delta),
                ]

            best_n, best_n_sum = s, current_sum
            for nb in neighbors:
                nb_sum = f(nb)
                if nb_sum > best_n_sum:
                    best_n, best_n_sum = nb, nb_sum

            if best_n_sum > current_sum:
                s, current_sum = best_n, best_n_sum
                consec += 1
                if consec >= 2:
                    delta *= 2
                    consec = 0
            else:
                delta /= 2
                consec = 0

        # Update global if improved
        if current_sum > best_cont_sum:
            best_cont_sum = current_sum
            if metric == "haversine":
                lat_c, lon_c = s
                dlat = lat_rad - lat_c
                dlon = lon_rad - lon_c
                a = (
                    np.sin(dlat / 2) ** 2
                    + np.cos(lat_rad) * np.cos(lat_c) * np.sin(dlon / 2) ** 2
                )
                d = 2 * np.arcsin(np.sqrt(a))
                best_cont_group = np.where(d <= thr)[0]
                best_center = (float(np.degrees(lat_c)),
                               float(np.degrees(lon_c)))
            else:
                dx = xs - s[0]
                dy = ys - s[1]
                best_cont_group = np.where(dx * dx + dy * dy <= thr)[0]
                best_center = invert_euclidean(s, lat0, lon0)

    # 5) Results
    return best_cont_group, best_cont_sum, best_center
