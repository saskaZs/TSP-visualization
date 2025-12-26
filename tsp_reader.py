"""
TSP File Reader Module
Handles reading and parsing TSP files in TSPLIB format
"""

import numpy as np


def read_tsp_file(filename: str) -> np.ndarray:
    """
    Reads TSP file in TSPLIB format and extracts city coordinates.
    
    The TSPLIB format typically contains:
    - Header information (NAME, TYPE, DIMENSION, etc.)
    - NODE_COORD_SECTION with city coordinates
    - EOF marker
    
    Args:
        filename: Path to the .tsp file
    
    Returns:
        numpy array of shape (n_cities, 2) with x, y coordinates
    """
    coordinates = []
    reading_coords = False
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Start reading coordinates when we hit this section
            if line.startswith('NODE_COORD_SECTION'):
                reading_coords = True
                continue
            
            # Stop reading at EOF
            if line == 'EOF' or line.startswith('EOF'):
                break
            
            # Parse coordinate lines
            if reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    # Format: node_id x_coord y_coord
                    x, y = float(parts[1]), float(parts[2])
                    coordinates.append([x, y])
    
    return np.array(coordinates)


def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Calculates Euclidean distance matrix between all cities.
    
    The distance matrix is symmetric where dist[i][j] represents
    the Euclidean distance from city i to city j.
    
    Args:
        coords: Array of city coordinates (n_cities x 2)
    
    Returns:
        Distance matrix (n_cities x n_cities)
    """

    n = len(coords)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Euclidean distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
                dist_matrix[i][j] = np.sqrt(
                    (coords[i][0] - coords[j][0])**2 + 
                    (coords[i][1] - coords[j][1])**2
                )
    
    return dist_matrix


def calculate_tour_length(tour: list, dist_matrix: np.ndarray) -> float:
    """
    Calculates total length of a tour.
    
    The tour is represented as a list of city indices.
    This function sums up all distances between consecutive cities
    and includes the distance back to the starting city.
    
    Args:
        tour: List of city indices representing the tour order
        dist_matrix: Pre-computed distance matrix
    
    Returns:
        Total tour length (sum of all edge distances)
    """
    
    length = 0
    n = len(tour)
    
    for i in range(n):
        # Add distance from current city to next city
        # Use modulo to wrap around to start city
        current_city = tour[i]
        next_city = tour[(i + 1) % n]
        length += dist_matrix[current_city][next_city]
    
    return length