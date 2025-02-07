# proxigraph/plot.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


###############################################################################
# Helper Functions for Colors from Images
###############################################################################

def load_image(image_path):
    """
    Load an image from a given path.

    Parameters:
        image_path (str): The full path to the image file.

    Returns:
        PIL.Image: Loaded image.
    """
    return Image.open(image_path)


def map_points_to_colors(positions_df, filename, args):
    """
    Map points to colors using an image.

    The function loads the image (from args.directory_map["colorfolder"] and the provided filename),
    then maps each node’s position to a color (based on the corresponding pixel value).

    Parameters:
        positions_df (pandas.DataFrame): DataFrame containing at least 'x', 'y', and 'node_ID'.
        filename (str): The image filename (e.g., 'colors.jpg').
        args: An object that contains configuration, including directory_map.

    Returns:
        dict: A dictionary mapping node_ID to an RGB tuple normalized to [0, 1].
    """
    image_folder = args.directory_map["colorfolder"]
    image_path = os.path.join(image_folder, filename)
    image = load_image(image_path).convert("RGB")
    pixels = np.array(image)
    image_width, image_height = image.size

    # Get the extent of the node positions.
    df_min_x, df_max_x = positions_df['x'].min(), positions_df['x'].max()
    df_min_y, df_max_y = positions_df['y'].min(), positions_df['y'].max()

    # Compute scaling factors from data space to image pixel coordinates.
    scale_x = image_width / (df_max_x - df_min_x) if df_max_x != df_min_x else 1
    scale_y = image_height / (df_max_y - df_min_y) if df_max_y != df_min_y else 1

    node_id_to_color = {}
    for idx, row in positions_df.iterrows():
        x_scaled = int((row['x'] - df_min_x) * scale_x)
        y_scaled = int((row['y'] - df_min_y) * scale_y)
        node_id = row['node_ID']
        if 0 <= x_scaled < image_width and 0 <= y_scaled < image_height:
            # Note: PIL images use (x, y) indexing where y is the row.
            color = tuple(pixels[y_scaled, x_scaled] / 255.0)  # Normalize to 0-1 range.
            node_id_to_color[node_id] = color
        else:
            # If out of bounds, default to black.
            node_id_to_color[node_id] = (0, 0, 0)
    return node_id_to_color


###############################################################################
# Node and Edge Plotting Functions
###############################################################################

def plot_nodes(ax, positions_df, args, color_mapping=None, colormap=None):
    """
    Plot nodes on a given axis.

    Parameters:
        ax: Matplotlib Axes object.
        positions_df (pandas.DataFrame): DataFrame with columns 'x', 'y' (and optionally 'z') and 'node_ID'.
        args: An object containing at least the attribute 'dim' (2 or 3).
        color_mapping (dict, optional): Mapping from node_ID to a color (tuple or color string).
            If provided, these colors are used.
        colormap (str, optional): If no color_mapping is provided, a matplotlib colormap is used to
            map the x-coordinate to a color.
    """
    if color_mapping is not None:
        # Map each node's ID to its corresponding color.
        colors = positions_df['node_ID'].map(color_mapping)
    elif colormap is not None:
        # Use a colormap based on the x-coordinate.
        norm = plt.Normalize(vmin=positions_df['x'].min(), vmax=positions_df['x'].max())
        cmap = plt.get_cmap(colormap)
        colors = positions_df['x'].apply(lambda x: cmap(norm(x)))
    else:
        colors = 'b'  # Default blue.

    if args.dim == 3:
        ax.scatter(positions_df['x'], positions_df['y'], positions_df['z'], c=colors)
    else:
        ax.scatter(positions_df['x'], positions_df['y'], c=colors)

    # Optionally, remove ticks and axes.
    ax.set_xticks([])
    ax.set_yticks([])
    if args.dim == 3:
        ax.set_zticks([])


def plot_edges(ax, positions_df, edges_df, args, edge_color='k', linewidth=0.5, alpha=0.8):
    """
    Plot edges on a given axis.

    Parameters:
        ax: Matplotlib Axes object.
        positions_df (pandas.DataFrame): DataFrame containing node positions.
        edges_df (pandas.DataFrame): DataFrame containing edge data with at least 'source' and 'target'.
        args: An object with attribute 'dim' (2 or 3).
        edge_color (str): Color for the edges.
        linewidth (float): Width of the edge lines.
        alpha (float): Transparency for the edges.
    """
    for _, row in edges_df.iterrows():
        source_id = row['source']
        target_id = row['target']
        source = positions_df.loc[positions_df['node_ID'] == source_id]
        target = positions_df.loc[positions_df['node_ID'] == target_id]
        if source.empty or target.empty:
            continue
        source = source.iloc[0]
        target = target.iloc[0]
        if args.dim == 3:
            ax.plot([source['x'], target['x']],
                    [source['y'], target['y']],
                    [source['z'], target['z']],
                    color=edge_color, linewidth=linewidth, alpha=alpha)
        else:
            ax.plot([source['x'], target['x']],
                    [source['y'], target['y']],
                    color=edge_color, linewidth=linewidth, alpha=alpha)


###############################################################################
# Main Function: Plotting Three Graph Modes in Subplots
###############################################################################

def plot_graphs_modes(args, positions_df, edges_dict, color_mapping=None, colormap='viridis', plot_edges_flag=True):
    """
    Generate a figure with three subplots corresponding to three different graph generation modes.

    Parameters:
        args: Configuration object that must include at least 'dim'.
        positions_df (pandas.DataFrame): DataFrame with node positions (columns 'x', 'y', 'node_ID', and optionally 'z').
        edges_dict (dict): Dictionary mapping mode names (str) to edges DataFrames.
            For example: {'knn': edges_knn_df, 'epsilon-ball': edges_eps_df, 'delaunay': edges_del_df}
        color_mapping (dict, optional): If provided, a dictionary mapping node_ID to color.
        colormap (str, optional): If no color_mapping is provided, the colormap (default 'viridis') will be used
            to assign colors based on the x-coordinate.
        plot_edges_flag (bool): If True, the edges are plotted on each subplot.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Create a figure with 1 row and 3 columns.
    fig, axs = plt.subplots(1, 3, figsize=(18, 6),
                            subplot_kw={'projection': '3d'} if args.dim == 3 else None)

    # Ensure axs is an iterable.
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    mode_labels = list(edges_dict.keys())

    for ax, mode in zip(axs, mode_labels):
        # Plot nodes using either image-based colors (if provided) or a default colormap.
        plot_nodes(ax, positions_df, args, color_mapping=color_mapping, colormap=colormap)
        # Plot edges if requested and if edge data is available.
        if plot_edges_flag and edges_dict.get(mode) is not None:
            plot_edges(ax, positions_df, edges_dict[mode], args)
        ax.set_title(mode)

    plt.tight_layout()
    return fig



def plot_decay_effects(distances, probabilities, edge_distances, quantile_scale):
    """
    Plot decay effects.

    Parameters:
        distances (array-like): Array of distances.
        probabilities (array-like): Decay probabilities.
        edge_distances (array-like): Distances of selected edges.
        quantile_scale (float): Quantile scale used.
    """
    # Scatter plot of distances vs probabilities
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, probabilities, alpha=0.5)
    plt.xlabel("Distance")
    plt.ylabel("Probability")
    plt.title(f"Decay Function (quantile_scale={quantile_scale})")
    plt.grid(True)
    plt.show()

    # Histogram of edge distances
    plt.figure(figsize=(10, 6))
    plt.hist(edge_distances, bins=50, alpha=0.75)
    plt.xlabel("Edge Distance")
    plt.ylabel("Frequency")
    plt.title("Histogram of Edge Distances")
    plt.show()


def plot_graph(positions, edges=None, config=None, ax=None):
    """
    Plot the point cloud (and optionally edges) of the graph.

    Parameters:
      positions : NumPy array of shape (N,2) or (N,3) representing node positions.
      edges     : Optional list of edge tuples (i, j). If provided, edges are drawn.
      config    : GraphConfig instance (used to choose colormap, dimension, etc.).
      ax        : Optional matplotlib Axes object. If provided, plotting occurs on this axis.

    Returns:
      ax : The matplotlib Axes object used for plotting.
    """
    # Create a new figure/axis if none is provided.
    if ax is None:
        if config is not None and config.dim == 3:
            # Create a 3D axis if in 3D mode.
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

    # Choose colors based on the x-coordinate.
    norm = plt.Normalize(vmin=positions[:, 0].min(), vmax=positions[:, 0].max())
    cmap = cm.get_cmap("viridis")
    colors = [cmap(norm(x)) for x in positions[:, 0]]

    # Plot nodes.
    if config is not None and config.dim == 3:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors)
    else:
        ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=40)

    # Plot edges if provided.
    if edges is not None:
        for i, j in edges:
            p1 = positions[i]
            p2 = positions[j]
            if config is not None and config.dim == 3:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="gray", lw=0.5)
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", lw=0.5)

    # Remove axis ticks for a cleaner look.
    ax.set_xticks([])
    ax.set_yticks([])
    if config is not None and config.dim == 3:
        try:
            ax.set_zticks([])
        except AttributeError:
            pass  # In case the provided axis isn't a 3D axis.

    return ax