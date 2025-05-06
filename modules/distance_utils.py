# ────────────────────────────────────────────────────────────────────────────────
# n×n matrix routines distances
# ────────────────────────────────────────────────────────────────────────────────
import numpy as np

# Earth radius in meters
R = 6_371_000.0

def haversine_matrix(latitudes, longitudes):
    """
    Compute the full pairwise Haversine distance matrix (in meters) between all points.
    
    Parameters
    ----------
    latitudes : array_like, shape (n,)
        Latitudes en grados decimales.
    longitudes : array_like, shape (n,)
        Longitudes en grados decimales.
    
    Returns
    -------
    D : ndarray, shape (n, n)
        Matriz de distancias en metros, donde D[i, j] es la distancia Haversine
        entre el punto i y el punto j.
    """
    # 1) Convertir a radianes
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)

    # 2) Diferencias por pares
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]

    # 3) Fórmula Haversine
    a = (
        np.sin(dlat / 2.0)**2
        + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2.0)**2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    # 4) Distancia
    return R * c

def euclidean_matrix(latitudes, longitudes):
    """
    Compute the full pairwise Euclidean distance matrix (in meters) 
    between all points, converting lat/lon to planar x/y.
    
    Parameters:
    - latitudes (np.ndarray): Vector of latitudes in decimal degrees.
    - longitudes (np.ndarray): Vector of longitudes in decimal degrees.

    Returns:
    - np.ndarray: Matrix of distances in meters (n x n), where n is the number of points. 
    """
    # Convert degrees to radians
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    mean_lat = np.mean(lat_rad)

    # Project lat/lon to local x/y plane
    x = R * lon_rad * np.cos(mean_lat)
    y = R * lat_rad

    # Compute pairwise Euclidean distance
    dx = x.reshape(-1, 1) - x.reshape(1, -1)
    dy = y.reshape(-1, 1) - y.reshape(1, -1)

    return np.sqrt(dx**2 + dy**2)