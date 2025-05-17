"""Synthetic policy generators for city and nationwide Spanish samples."""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import io
from shapely.vectorized import contains as vectorized_contains
from typing import Optional
from numpy.typing import NDArray


# —————————————————————————————————————————————————————————————————————
# Helpers
# —————————————————————————————————————————————————————————————————————
def _assign_insured_sums(
    n: int,
    high_value_ratio: float,
    low: int = 1_000,
    high: int = 100_000,
    hv_low: int = 500_000,
    hv_high: int = 2_500_000,
    seed: Optional[int] = None,
) -> NDArray[np.int_]:
    """
    Generate an array of insured sums with a mix of low and high values.

    Produces `n` random insured sums uniformly between `low` and `high`,
    then elevates a fraction (`high_value_ratio`) of them to high-value
    ranges between `hv_low` and `hv_high`.

    Parameters:
    - n (int): Total number of policies.
    - high_value_ratio (float): Fraction of policies to assign high
      insured values.
    - low (int): Minimum insured sum for regular policies.
    - high (int): Maximum insured sum for regular policies.
    - hv_low (int): Minimum insured sum for high-value policies.
    - hv_high (int): Maximum insured sum for high-value policies.
    - seed (Optional[int]): Optional random seed for reproducibility.

    Returns:
    - NDArray[np.int_]: Array of length `n` with assigned insured sums.
    """
    if seed is not None:
        np.random.seed(seed + 1)
    insured = np.random.randint(low, high, size=n)
    n_high = int(n * high_value_ratio)
    if n_high > 0:
        hi_idx = np.random.choice(n, n_high, replace=False)
        insured[hi_idx] = np.random.randint(hv_low, hv_high, size=n_high)
    return insured


def _save_df(df: pd.DataFrame, output_path: str) -> None:
    """Save a DataFrame to CSV, creating directories as needed.

    Ensures the directory for `output_path` exists, then writes `df`
    to a CSV file without the index.

    Parameters:
    - df (pd.DataFrame): DataFrame to save.
    - output_path (str): File path where the CSV will be written.

    Returns:
    - None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


# —————————————————————————————————————————————————————————————————————
# 1) City-zones sampler
# —————————————————————————————————————————————————————————————————————
ZONES = {
    "madrid": {"lat_range": (40.40, 40.45), "lon_range": (-3.72, -3.67)},
    "valencia": {"lat_range": (39.45, 39.50), "lon_range": (-0.40, -0.35)},
    "barcelona": {"lat_range": (41.36, 41.41), "lon_range": (2.10, 2.16)},
}


def generate_synthetic_policies(
    output_path: str,
    n_policies: int = 1000,
    zones: list[str] = ["madrid"],
    seed: Optional[int] = None,
    high_value_ratio: float = 0.01,
) -> None:
    """Generate synthetic insurance policies within specified zones.

    Distributes `n_policies` across given Spanish zones, samples random
    lat/lon within each zone’s bounding box, assigns insured sums
    with a fraction of high-value policies, and writes to CSV.

    Parameters:
    - output_path (str): Destination file path for the CSV.
    - n_policies (int): Total number of policies to generate (default 1000).
    - zones (list[str]): List of zone names (keys in ZONES) to sample from.
    - seed (Optional[int]): Random seed for reproducibility (default None).
    - high_value_ratio (float): Fraction of policies to assign high values
      (default 0.01).

    Returns:
    - None
    """
    if seed is not None:
        np.random.seed(seed)
    if isinstance(zones, str):
        zones = [zones]
    unknown = [z for z in zones if z not in ZONES]
    if unknown:
        raise ValueError(f"Unknown zones: {unknown}")
    n_zones = len(zones)
    base, rem = divmod(n_policies, n_zones)

    rows = []
    next_id = 10000
    for i, zone in enumerate(zones):
        count = base + (1 if i < rem else 0)
        lat_min, lat_max = ZONES[zone]["lat_range"]
        lon_min, lon_max = ZONES[zone]["lon_range"]
        ids = np.arange(next_id, next_id + count)
        lats = np.random.uniform(lat_min, lat_max, size=count)
        lons = np.random.uniform(lon_min, lon_max, size=count)
        sums = _assign_insured_sums(count, high_value_ratio, seed=seed)
        df = pd.DataFrame(
            {
                "n_policy": ids,
                "lat": lats,
                "lon": lons,
                "insured_sum": sums,
                "zone": zone,
            }
        )
        rows.append(df)
        next_id += count

    full_df = pd.concat(rows, ignore_index=True)
    _save_df(full_df, output_path)


# —————————————————————————————————————————————————————————————————————
# 2) Whole-Spain sampler
# —————————————————————————————————————————————————————————————————————
def generate_synthetic_spanish_policies(
    output_path: str,
    n_policies: int = 1000,
    seed: Optional[int] = None,
    high_value_ratio: float = 0.01,
) -> None:
    """
    Generate a synthetic CSV of random insurance policies across Spain.

    Samples uniformly random points within Spain’s geographic boundary,
    assigns insured sums with a small fraction of high-value policies,
    and writes the resulting DataFrame to the given path.

    Parameters:
    - output_path (str): File path where the CSV will be saved.
    - n_policies (int): Total number of policies to generate (default 1000).
    - seed (Optional[int]): Random seed for reproducibility (default None).
    - high_value_ratio (float): Fraction of policies to assign high values
      (default 0.01).

    Returns:
    - None
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) Fetch the raw GeoJSON with requests
    url = (
        "https://raw.githubusercontent.com/ufoe/"
        "d3js-geojson/master/Spain.json"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    geojson_str = resp.text

    # 2) Read into GeoDataFrame from an in‐memory string
    gdf_spain = gpd.read_file(io.StringIO(geojson_str))
    spain_geom = gdf_spain.geometry.unary_union

    # 3) Bounding box for quick rejection
    minx, miny, maxx, maxy = spain_geom.bounds

    # 4) Batch‐sample random points & keep those inside Spain
    lons_list: list[float] = []
    lats_list: list[float] = []
    oversample = 1.3
    while len(lons_list) < n_policies:
        need = n_policies - len(lons_list)
        batch = int(need * oversample)
        xs = np.random.uniform(minx, maxx, size=batch)
        ys = np.random.uniform(miny, maxy, size=batch)
        mask = vectorized_contains(spain_geom, xs, ys)
        xs_in = xs[mask]
        ys_in = ys[mask]
        take = min(need, xs_in.size)
        lons_list.extend(xs_in[:take].tolist())
        lats_list.extend(ys_in[:take].tolist())

    lons: NDArray[np.float64] = np.array(lons_list[:n_policies])
    lats: NDArray[np.float64] = np.array(lats_list[:n_policies])

    # 5) Assign insured_sums
    sums = _assign_insured_sums(n_policies, high_value_ratio, seed=seed)

    # 6) Build DataFrame and save
    df = pd.DataFrame(
        {
            "n_policy": np.arange(1, n_policies + 1),
            "lat": lats,
            "lon": lons,
            "insured_sum": sums,
            "zone": "spain",
        }
    )
    _save_df(df, output_path)
