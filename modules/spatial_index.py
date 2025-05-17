"""Coordinate conversions and BallTree helper functions for new algorithms."""
import numpy as np
from sklearn.neighbors import BallTree
from typing import Tuple, Literal, List
from numpy.typing import NDArray
from scipy.sparse import coo_matrix

# Earth radius in meters
R = 6_371_000.0


# ────────────────────────────────────────────────────────────────────────────────
# Coordinate conversions and BallTree helpers
# ────────────────────────────────────────────────────────────────────────────────
def haversine_coords(latitudes: NDArray[np.float64],
                     longitudes: NDArray[np.float64]
                     ) -> NDArray[np.float64]:
    """Haversine coordinates function.

    Convert decimal-degree lat/lon arrays into a Nx2 array of
    (lat_rad, lon_rad) for use with BallTree(metric='haversine').

    This function prepares geographic coordinates by converting
    from degrees to radians.

    Parameters:
    - latitudes (np.ndarray): Vector of latitudes in decimal degrees.
    - longitudes (np.ndarray): Vector of longitudes in decimal degrees.

    Returns:
    - np.ndarray: Array of shape (n, 2) where each row is
    (latitude_rad, longitude_rad).
    """
    return np.column_stack((np.deg2rad(latitudes), np.deg2rad(longitudes)))


def euclidean_coords(latitudes: NDArray[np.float64],
                     longitudes: NDArray[np.float64]
                     ) -> NDArray[np.float64]:
    """Euclidean coordinates function.

    Project decimal-degree lat/lon into a local planar x/y coordinate
    system (meters) for use with BallTree(metric='euclidean').

    Uses an equirectangular approximation centered at the mean latitude
    to minimize distortion over small areas.

    Parameters:
    - latitudes (np.ndarray): Vector of latitudes in decimal degrees.
    - longitudes (np.ndarray): Vector of longitudes in decimal degrees.

    Returns:
    - np.ndarray: Array of shape (n, 2) where each row is (x_meters, y_meters).
    """
    lat_r = np.deg2rad(latitudes)
    lon_r = np.deg2rad(longitudes)
    lat0 = lat_r.mean()
    lon0 = lon_r.mean()
    x = R * (lon_r - lon0) * np.cos(lat0)
    y = R * (lat_r - lat0)
    return np.column_stack((x, y))


def invert_euclidean(
    point_xy: Tuple[float, float], lat0: float, lon0: float
) -> Tuple[float, float]:
    """
    Convert planar x/y offsets back to latitude and longitude.

    Given a point's local planar coordinates (meters) and a reference
    latitude/longitude in radians, compute the geographic coordinates.

    Parameters:
    - point_xy (Tuple[float, float]): The (x, y) offsets in meters from
      the reference point.
    - lat0 (float): Reference latitude in radians.
    - lon0 (float): Reference longitude in radians.

    Returns:
    - Tuple[float, float]: The (latitude, longitude) in decimal degrees.
    """
    x_m, y_m = point_xy
    lat_c_rad = y_m / R + lat0
    lon_c_rad = x_m / (R * np.cos(lat0)) + lon0
    return float(np.degrees(lat_c_rad)), float(np.degrees(lon_c_rad))


def build_balltree(coords: NDArray[np.float64],
                   metric: str) -> BallTree:
    """
    Construct a BallTree from coordinate data with the specified metric.

    Parameters:
    - coords (np.ndarray): Array of shape (n, 2) of prepared coordinates
      (radians for 'haversine', meters for 'euclidean').
    - metric (str): Distance metric, must be 'haversine' or 'euclidean'.

    Returns:
    - BallTree: Fitted BallTree for neighbor queries.

    Raises:
    - ValueError: If metric is not one of the supported options.
    """
    if metric not in ("haversine", "euclidean"):
        raise ValueError("metric must be 'haversine' or 'euclidean'")
    return BallTree(coords, metric=metric)


def tree_radius(radius_m: float, metric: str) -> float:
    """Tree radius function.

    Convert a radius in meters into the units expected by
    BallTree.query_radius.

    For 'haversine', the tree expects radius in radians (fraction of
    Earth's radius).
    For 'euclidean', the tree expects radius in the same linear units (meters).

    Parameters:
    - radius_m (float): Radius in meters.
    - metric (str): Distance metric, 'haversine' or 'euclidean'.

    Returns:
    - float: Radius in tree units (radians or meters).
    """
    if metric == "haversine":
        return radius_m / R
    elif metric == "euclidean":
        return radius_m
    else:
        raise ValueError("metric must be 'haversine' or 'euclidean'")


def create_grid_around_point(center_point: Tuple[float, float],
                             radius: float,
                             step: float,
                             metric: Literal["haversine",
                                             "euclidean"] = "haversine"
                             ) -> NDArray[np.float64]:
    """
    Create a fine grid around a center point.

    Parameters:
    - center_point: tuple or array (lat_rad, lon_rad) for haversine
    or (x, y) for euclidean
    - radius: radius in tree units (radians for haversine, meters
    for euclidean)
    - step: grid step size in meters
    - metric: 'haversine' or 'euclidean'

    Returns:
    - np.ndarray: Grid points as a 2D array
    """
    if metric == "haversine":
        lat0, lon0 = center_point
        dlat = step / R
        dlon = step / (R * np.cos(lat0))
        lat_grid = np.arange(lat0 - radius, lat0 + radius + dlat, dlat)
        lon_grid = np.arange(
            lon0 - radius / np.cos(lat0), lon0
            + radius / np.cos(lat0) + dlon, dlon
        )
        g_lat, g_lon = np.meshgrid(lat_grid, lon_grid)
        return np.column_stack((g_lat.ravel(), g_lon.ravel()))
    else:
        x0, y0 = center_point
        d = step
        xs = np.arange(x0 - radius, x0 + radius + d, d)
        ys = np.arange(y0 - radius, y0 + radius + d, d)
        gx, gy = np.meshgrid(xs, ys)
        return np.column_stack((gx.ravel(), gy.ravel()))


def create_incidence_matrix(neighbors_list: List[NDArray[np.int_]],
                            n_points: int, n_centers: int
                            ) -> coo_matrix:
    """
    Create a sparse incidence matrix from a list of neighbors.

    Parameters:
    - neighbors_list: List of arrays where each array contains
    indices of neighbors
    - n_points: Number of data points
    - n_centers: Number of center points (typically len(neighbors_list))

    Returns:
    - scipy.sparse.coo_matrix: Sparse incidence matrix
    """
    rows, cols = [], []
    for idx, nbr in enumerate(neighbors_list):
        rows.extend([idx] * len(nbr))
        cols.extend(nbr.tolist())

    data = np.ones(len(rows), dtype=int)
    return coo_matrix((data, (rows, cols)), shape=(n_centers, n_points))
