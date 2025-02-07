# proxigraph/graphs.py

import numpy as np
import math
from collections import defaultdict
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


def compute_knn_graph(positions, k):
    """
    Compute the k-nearest neighbors graph.

    Parameters:
        positions : array-like of node positions.
        k         : int, number of nearest neighbors.

    Returns:
        distances : array of distances (excluding self).
        indices   : array of neighbor indices (excluding self).
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(positions)
    distances, indices = nbrs.kneighbors(positions)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    return distances, indices


def compute_epsilon_ball_graph(positions, radius):
    """
    Compute the epsilon-ball graph.

    Parameters:
        positions : array-like of node positions.
        radius    : float, radius within which neighbors are considered.

    Returns:
        distances : list of arrays with distances for each node.
        indices   : list of arrays with neighbor indices for each node.
    """
    nbrs = NearestNeighbors(radius=radius).fit(positions)
    distances, indices = nbrs.radius_neighbors(positions, sort_results=True)
    distances = [d[1:] if len(d) > 0 else d for d in distances]
    indices = [i[1:] if len(i) > 0 else i for i in indices]
    return distances, indices


def get_delaunay_neighbors(positions):
    """
    Compute neighbors using Delaunay triangulation.

    Parameters:
        positions : array-like of node positions.

    Returns:
        distances : list of arrays with distances to each neighbor.
        indices   : list of arrays with neighbor indices.
    """
    tess = Delaunay(positions)
    neighbors = defaultdict(set)
    for simplex in tess.simplices:
        for idx in simplex:
            others = set(simplex)
            others.remove(idx)
            neighbors[idx] = neighbors[idx].union(others)
    indices = []
    distances = []
    for i in range(len(positions)):
        neigh = sorted(list(neighbors[i]))
        indices.append(neigh)
        dist = [math.dist(positions[i], positions[j]) for j in neigh]
        distances.append(dist)
    return distances, indices


def compute_proximity_decay_graph(positions, decay_mode='decay_exp',
                                  quantile_scale=0.05, power_law_exp=4):
    """
    Compute a proximity decay graph.

    Parameters:
        positions     : array-like of node positions.
        decay_mode    : str, 'decay_exp' (exponential) or 'power_law'.
        quantile_scale: float, quantile for scaling distance.
        power_law_exp : int, exponent for power law (if used).

    Returns:
        distances : None (distances not computed explicitly here).
        indices   : list representing the adjacency list of neighbors.
    """

    def decay_exp(distance, scale, exponent):
        return np.exp(- (distance / scale) ** exponent)

    raw_distances = squareform(pdist(positions))
    distances_flat = raw_distances.flatten()
    distance_scale = np.quantile(distances_flat, quantile_scale)

    if decay_mode == 'power_law':
        probabilities = 1 / (np.power(raw_distances / distance_scale, power_law_exp) + 0.001)
    else:
        probabilities = decay_exp(raw_distances, scale=distance_scale, exponent=2)

    probabilities /= np.max(probabilities)
    np.fill_diagonal(probabilities, 0)

    n = len(positions)
    indices_pairs = np.transpose(np.triu_indices(n, k=1))
    selected = []
    for i, j in indices_pairs:
        if np.random.rand() < probabilities[i, j]:
            selected.append((i, j))

    adjacency_list = [[] for _ in range(n)]
    for i, j in selected:
        adjacency_list[i].append(j)
        adjacency_list[j].append(i)

    return None, adjacency_list


# --- Extra Modalities --- #


def knn_bipartite(positions, k, ratio=2):
    """
    Compute a bipartite k-nearest neighbors graph.

    Parameters:
        positions : array-like of node positions.
        k         : int, number of neighbors (per node) in the bipartite graph.
        ratio     : int, partition ratio (e.g., ratio=2 gives 50:50 split).

    Returns:
        distances : array of distances for each node.
        indices   : array of neighbor indices for each node.
    """
    positions = np.array(positions)
    n = len(positions)
    half = n // ratio
    bottom = positions[:half]
    top = positions[half:]

    nbrs_bottom = NearestNeighbors(n_neighbors=k).fit(top)
    distances_bottom, indices_bottom = nbrs_bottom.kneighbors(bottom)
    indices_bottom += half  # offset for top indices

    nbrs_top = NearestNeighbors(n_neighbors=k).fit(bottom)
    distances_top, indices_top = nbrs_top.kneighbors(top)

    distances = np.zeros((n, k))
    indices = np.zeros((n, k), dtype=int)
    distances[:half] = distances_bottom
    distances[half:] = distances_top
    indices[:half] = indices_bottom
    indices[half:] = indices_top
    return distances, indices


def epsilon_bipartite(positions, radius, ratio=2):
    """
    Compute a bipartite epsilon-ball graph.

    Parameters:
        positions : array-like of node positions.
        radius    : float, radius within which neighbors are considered.
        ratio     : int, partition ratio (e.g., ratio=2 gives 50:50 split).

    Returns:
        distances : list of arrays with distances for each node.
        indices   : list of arrays with neighbor indices for each node.
    """
    positions = np.array(positions)
    n = len(positions)
    half = n // ratio
    bottom = positions[:half]
    top = positions[half:]

    nbrs_bottom = NearestNeighbors(radius=radius).fit(top)
    distances_bottom, indices_bottom = nbrs_bottom.radius_neighbors(bottom, sort_results=True)
    # Offset indices for bottom part:
    indices_bottom = [ind + half for ind in indices_bottom]

    nbrs_top = NearestNeighbors(radius=radius).fit(bottom)
    distances_top, indices_top = nbrs_top.radius_neighbors(top, sort_results=True)

    distances = [None] * n
    indices = [None] * n
    for i in range(half):
        distances[i] = distances_bottom[i]
        indices[i] = indices_bottom[i]
    for i in range(half, n):
        distances[i] = distances_top[i - half]
        indices[i] = indices_top[i - half]
    return distances, indices


def get_delaunay_neighbors_corrected(positions):
    """
    Compute a corrected Delaunay graph by removing overly long edges and ensuring connectivity.

    Parameters:
        positions : array-like of node positions.

    Returns:
        distances : list of arrays with distances for each node.
        indices   : list of arrays with neighbor indices for each node.
    """
    tess = Delaunay(positions)
    neighbors = defaultdict(set)
    for simplex in tess.simplices:
        for idx in simplex:
            neighbors[idx].update(set(simplex) - {idx})

    # Build edge list with distances
    n = len(positions)
    edges = []
    for i in range(n):
        for j in neighbors[i]:
            if i < j:
                d = math.dist(positions[i], positions[j])
                edges.append((i, j, d))
    edges = np.array(edges, dtype=[('i', int), ('j', int), ('d', float)])
    edges_sorted = np.sort(edges, order='d')

    # Determine cutoff (e.g., 95th percentile)
    cutoff = np.percentile(edges_sorted['d'], 95)
    valid_edges = edges_sorted[edges_sorted['d'] <= cutoff]

    # Build graph matrix and ensure connectivity
    graph_matrix = csr_matrix((np.ones(len(valid_edges)),
                               (valid_edges['i'], valid_edges['j'])), shape=(n, n))
    n_components, _ = connected_components(graph_matrix)
    if n_components > 1:
        edges_to_keep = list(valid_edges)
        for edge in edges_sorted:
            if edge['d'] > cutoff:
                edges_to_keep.append(edge)
                row = [e['i'] for e in edges_to_keep]
                col = [e['j'] for e in edges_to_keep]
                graph_matrix = csr_matrix((np.ones(len(edges_to_keep)), (row, col)),
                                          shape=(n, n))
                n_components, _ = connected_components(graph_matrix)
                if n_components == 1:
                    break
    else:
        edges_to_keep = valid_edges

    # Build adjacency list
    adj_indices = [[] for _ in range(n)]
    adj_distances = [[] for _ in range(n)]
    for edge in edges_to_keep:
        i, j, d = edge['i'], edge['j'], edge['d']
        adj_indices[i].append(j)
        adj_indices[j].append(i)
        adj_distances[i].append(d)
        adj_distances[j].append(d)

    adj_indices = [np.array(neigh) for neigh in adj_indices]
    adj_distances = [np.array(dists) for dists in adj_distances]

    return adj_distances, adj_indices


def compute_lattice(args, positions):
    """
    Compute a lattice graph (square for 2D, cubic for 3D).

    Parameters:
        args      : An object with attribute 'dim'.
        positions : Array-like of node positions (assumed to lie on a lattice).

    Returns:
        distances : Array of distances (excluding self).
        indices   : Array of neighbor indices (excluding self).
    """
    n_neighbors = 4 if args.dim == 2 else 6
    distances, indices = compute_knn_graph(positions, k=n_neighbors + 1)
    return distances, indices

def compute_epsilon_ball_radius(density, intended_degree, dim, base_proximity_mode):
    """
    Compute the radius for an epsilon-ball graph based on the density of points,
    the intended average degree, and the dimension.

    Parameters:
        density             (float): Density of points in the region.
        intended_degree     (int)  : The target average degree.
        dim                 (int)  : Dimension of the space (2 or 3).
        base_proximity_mode (str)  : Base mode (e.g., "epsilon-ball" or "epsilon_bipartite").

    Returns:
        radius (float): The computed radius.
    """
    if dim == 2:
        radius_coefficient = np.pi   # Area of a unit circle
    elif dim == 3:
        radius_coefficient = (4 / 3) * np.pi  # Volume of a unit sphere
    else:
        raise ValueError("Dimension must be 2 or 3.")

    # We add 1 to the intended degree (or double it for bipartite cases) to account for the
    # fact that the origin point is not included.
    if base_proximity_mode == "epsilon_bipartite":
        effective_degree = 2 * intended_degree + 1
    else:
        effective_degree = intended_degree + 1

    # Compute and return the radius.
    return ((effective_degree) / (radius_coefficient * density)) ** (1 / dim)




def compute_density(positions, dim):
    num_points = len(positions)
    bbox_min = np.min(positions, axis=0)
    bbox_max = np.max(positions, axis=0)
    extent = bbox_max - bbox_min
    if dim == 2:
        area = extent[0] * extent[1]
        return num_points / area if area > 0 else num_points
    elif dim == 3:
        volume = extent[0] * extent[1] * extent[2]
        return num_points / volume if volume > 0 else num_points
    else:
        raise ValueError("Dimension must be 2 or 3.")