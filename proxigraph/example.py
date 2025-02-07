# example.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

from proxigraph.config import GraphConfig
from proxigraph.core import ProximityGraph
from proxigraph.plot import plot_graph


def minimal_plot_example():
    config = GraphConfig()
    pg = ProximityGraph(config=config)
    positions = np.array(pg.generate_positions())
    edges = pg.compute_graph()
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_graph(positions, edges=edges, config=config, ax=ax)
    plt.show()


def minimal_edge_list_example():
    config = GraphConfig()
    pg = ProximityGraph(config=config)
    positions = pg.generate_positions()
    edges = pg.compute_graph()

    # Get edge list as DataFrame.
    edge_df = pg.get_edge_list(as_dataframe=True)
    print("Edge List (DataFrame):")
    print(edge_df.head())

    # Get positions as DataFrame.
    pos_df = pg.get_positions_df()
    print("Positions DataFrame:")
    print(pos_df.head())
    # Get NetworkX graph.
    G = pg.get_networkx_graph()
    print("NetworkX Graph:")
    print(G)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_graph(positions, edges=edges, config=config, ax=ax)
    plt.show()


def compare_proximity_modes_example():
    modes = ["delaunay_corrected", "epsilon-ball", "knn_bipartite"]
    # All examples here are 2D, so we can use a standard 2D subplot grid.
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, mode in zip(axs, modes):
        config = GraphConfig(proximity_mode=mode)
        pg = ProximityGraph(config=config)
        positions = pg.generate_positions()
        pg.compute_graph()
        plot_graph(positions, edges=pg.edge_list, config=config, ax=ax)
        ax.set_title(mode)
    plt.suptitle("Comparison of Proximity Modes")
    plt.tight_layout()
    plt.show()


def compare_point_modes_example():
    # Here we compare three different point modes:
    #   1. "circle" (2D uniform)
    #   2. "square" with proximity mode "lattice" (3D)
    #   3. "star" (image-based, 2D)
    fig = plt.figure(figsize=(18, 6))
    # First subplot: 2D circle
    ax1 = fig.add_subplot(131)  # default 2D axis
    config_circle = GraphConfig(
        dim=2,
        num_points=1000,
        L=10,
        point_mode="circle",
        proximity_mode="epsilon-ball",
        intended_av_degree=10,
        directory_map={"colorfolder": "./colors", "shapes": "./proxigraph/shapes"},
        density_anomalies=False
    )
    pg_circle = ProximityGraph(config=config_circle)
    positions_circle = pg_circle.generate_positions()
    pg_circle.compute_graph()
    plot_graph(positions_circle, edges=pg_circle.edge_list, config=config_circle, ax=ax1)
    ax1.set_title("Circle - Delaunay - Uniform - 2D")

    # Second subplot: 3D square lattice
    ax2 = fig.add_subplot(133, projection='3d')
    config_square = GraphConfig(
        dim=3,
        num_points=1000,
        L=10,
        point_mode="square",
        proximity_mode="lattice",
        intended_av_degree=6,
        directory_map={"colorfolder": "./colors", "shapes": "./proxigraph/shapes"},
        density_anomalies=False
    )
    pg_square = ProximityGraph(config=config_square)
    positions_square = pg_square.generate_positions()
    pg_square.compute_graph()
    plot_graph(positions_square, edges=pg_square.edge_list, config=config_square, ax=ax2)
    ax2.set_title("Cube - Lattice - Uniform - 3D")

    # Third subplot: 2D star (image-based)
    ax3 = fig.add_subplot(132)
    config_star = GraphConfig(
        dim=2,
        num_points=1000,
        L=10,
        point_mode="star",
        proximity_mode="knn",
        intended_av_degree=10,
        density_anomalies=True,
    )
    pg_star = ProximityGraph(config=config_star)
    positions_star = pg_star.generate_positions()
    pg_star.compute_graph()
    plot_graph(positions_star, edges=pg_star.edge_list, config=config_star, ax=ax3)
    ax3.set_title("Star - kNN - Not Uniform - 2D")

    plt.suptitle("ProxiGraph Examples")
    plt.tight_layout()
    plt.show()


def compare_density_anomalies_example():
    # Compare two configurations for "circle" point mode in 2D,
    # one with density anomalies and one without.
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Without density anomalies
    config_uniform = GraphConfig(
        dim=2,
        num_points=1000,
        L=10,
        point_mode="circle",
        proximity_mode="epsilon-ball",
        intended_av_degree=10,
        directory_map={"colorfolder": "./colors", "shapes": "./proxigraph/shapes"},
        density_anomalies=False
    )
    pg_uniform = ProximityGraph(config=config_uniform)
    positions_uniform = pg_uniform.generate_positions()
    pg_uniform.compute_graph()
    plot_graph(positions_uniform, edges=pg_uniform.edge_list, config=config_uniform, ax=axs[0])
    axs[0].set_title("Circle without Density Anomalies")

    # With density anomalies (using the shapes image for uniform modes)
    config_anomaly = GraphConfig(
        dim=2,
        num_points=1000,
        L=10,
        point_mode="circle",
        proximity_mode="epsilon-ball",
        intended_av_degree=10,
        directory_map={"colorfolder": "./colors", "shapes": "./proxigraph/shapes"},
        density_anomalies=True
    )
    pg_anomaly = ProximityGraph(config=config_anomaly)
    positions_anomaly = pg_anomaly.generate_positions()
    pg_anomaly.compute_graph()
    plot_graph(positions_anomaly, edges=pg_anomaly.edge_list, config=config_anomaly, ax=axs[1])
    axs[1].set_title("Circle with Density Anomalies")

    plt.suptitle("Comparison of Density Anomalies")
    plt.tight_layout()
    plt.show()


def main():
    print("Running point modes comparison...")
    compare_point_modes_example()

    ### More examples:

    # print("Running minimal edge list example...")
    # minimal_edge_list_example()
    # print("Running minimal plot example...")
    # minimal_plot_example()
    # print("Running proximity modes comparison...")
    # compare_proximity_modes_example()
    # print("Running density anomalies comparison...")
    # compare_density_anomalies_example()


if __name__ == '__main__':
    main()
