# proxigraph/utils.py

import numpy as np
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch


def sort_points_by_distance_to_centroid(points):
    """
    Sort points by their distance to the centroid.

    Parameters:
        points: Array-like of points.

    Returns:
        Sorted NumPy array of points.
    """
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    sorted_indices = np.argsort(distances)
    return points[sorted_indices]




def distribute_weights(num_edges, total_weight, max_weight):
    """
    Randomly distribute weights among edges such that the sum equals total_weight.

    Parameters:
        num_edges (int): Number of edges.
        total_weight (int): Total weight to distribute.
        max_weight (int): Maximum weight per edge.

    Returns:
        NumPy array of weights.
    """
    weights = np.random.randint(1, max_weight + 1, size=num_edges)
    while sum(weights) != total_weight:
        diff = sum(weights) - total_weight
        for i in range(num_edges):
            if diff == 0:
                break
            if diff > 0 and weights[i] > 1:
                adjustment = min(weights[i] - 1, diff)
                weights[i] -= adjustment
                diff -= adjustment
            elif diff < 0:
                adjustment = min(max_weight - weights[i], -diff)
                weights[i] += adjustment
                diff += adjustment
    return weights


def add_weights_to_false_edges(edge_df, false_edge_ids, false_edges_count, max_weight):
    """
    Add weights to false edges in an edge DataFrame.

    Parameters:
        edge_df (pandas.DataFrame): DataFrame with columns ['source', 'target', 'weight'].
        false_edge_ids (list): List of edge tuples to be weighted.
        false_edges_count (int): Number of false edges.
        max_weight (int): Maximum weight per edge.

    Returns:
        Updated pandas DataFrame.
    """
    if 'weight' not in edge_df.columns:
        raise ValueError("Column 'weight' missing from DataFrame")

    total_weight = (false_edges_count * max_weight) // 3
    weights = distribute_weights(false_edges_count, total_weight, max_weight)

    for i, edge in enumerate(false_edge_ids):
        source, target = edge
        mask = (edge_df['source'] == source) & (edge_df['target'] == target)
        edge_df.loc[mask, 'weight'] = weights[i]
    return edge_df
