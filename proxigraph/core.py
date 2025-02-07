# proxigraph/core.py

import networkx as nx
import pandas as pd
from points import (generate_random_points,
                     generate_random_points_in_circle_or_sphere,
                     generate_points_from_image,
                     generate_points_from_image_with_anomalies,
                     generate_square_lattice,
                    )
from graphs import *
from importlib import resources
from proxigraph.config import GraphConfig

class ProximityGraph:
    """
    Main API class to generate a proximity graph.
    """

    def __init__(self, config: GraphConfig):
        self.config = config
        self.dim = config.dim
        self.num_points = config.num_points
        self.L = config.L
        self.point_mode = config.point_mode
        self.proximity_mode = config.proximity_mode
        self.intended_av_degree = config.intended_av_degree
        self.directory_map = config.directory_map
        self.color_mapping = config.color_mapping
        self.density_anomalies = config.density_anomalies
        self.positions = None
        self.edge_list = None

    def generate_positions(self):
        """
        Generate positions based on the chosen point_mode.

        Special Cases:
          - If self.proximity_mode is "lattice", a square lattice of points is generated.
          - If self.density_anomalies is True and self.dim == 2:
                For uniform modes ("square" or "circle"), an image from the "shapes" folder is used.
                For image-based modes (e.g. 'triangle', 'ring', etc.), the image path must be provided in self.extra_args.

        Returns:
            positions: A NumPy array of generated node positions.
        """
        # Special case: lattice mode
        if self.proximity_mode == "lattice":
            self.positions = np.array(generate_square_lattice(self))
            self.num_points = self.positions.shape[0]
            return self.positions

        # Handle 3D cases
        if self.dim == 3:
            if self.density_anomalies:
                raise ValueError("Density anomalies are not supported in 3D.")
            if self.point_mode == "square":
                self.positions = np.array(generate_random_points(self.num_points, self.L, self.dim))
            elif self.point_mode == "circle":
                self.positions = np.array(generate_random_points_in_circle_or_sphere(self.num_points, self.L, self.dim))
            else:
                raise ValueError("Invalid point mode for 3D.")

        # Handle 2D cases
        elif self.dim == 2:
            # Modes that use a dedicated shape image
            image_modes = {'triangle', 'ring', 'star', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

            if self.point_mode in ["square", "circle"]:
                if self.density_anomalies:
                    # Use an image from the "shapes" folder based on the point mode.
                    # For example, "square.png" for "square" and "circle.png" for "circle".
                    shape_filename = f"{self.point_mode}.png"
                    # Use importlib.resources to obtain a path to the resource
                    with resources.path("proxigraph.shapes", shape_filename) as image_path:
                        self.positions = np.array(
                            generate_points_from_image_with_anomalies(self.num_points, str(image_path)))
                else:
                    # Uniform generation without anomalies.
                    if self.point_mode == "square":
                        self.positions = np.array(generate_random_points(self.num_points, self.L, self.dim))
                    elif self.point_mode == "circle":
                        self.positions = np.array(
                            generate_random_points_in_circle_or_sphere(self.num_points, self.L, self.dim))
            elif self.point_mode in image_modes:
                shape_filename = f"{self.point_mode}.png"
                with resources.path("proxigraph.shapes", shape_filename) as image_path:
                    if self.density_anomalies:
                        self.positions = np.array(generate_points_from_image_with_anomalies(self.num_points,
                                                                                            str(image_path)))
                    else:
                        self.positions = np.array(generate_points_from_image(self.num_points,
                                                                             str(image_path)))
            else:
                raise ValueError("Invalid point mode for 2D.")
        else:
            raise ValueError("Dimension must be 2 or 3.")

        return self.positions



    def compute_graph(self):
        """
        Compute the proximity graph according to the selected mode.

        Supported modes:
          - "knn": Standard k-nearest neighbors.
          - "epsilon-ball": All points within a given radius.
          - "delaunay": Neighbors based on Delaunay triangulation.
          - "delaunay_corrected": Delaunay triangulation with long-edge filtering and reconnection.
          - "distance_decay": A graph whose edge probabilities decay with distance.
          - "knn_bipartite": k-NN computed across a bipartition of the node set.
          - "epsilon_bipartite": Epsilon-ball search across a bipartition.
          - "lattice": A lattice graph (special mode; assumes positions lie on a grid).

        Returns:
            edge_list: A list of undirected edge tuples (i, j).
        """
        if self.positions is None:
            self.generate_positions()

        if self.proximity_mode == "knn":
            distances, indices = compute_knn_graph(self.positions, self.intended_av_degree)

        elif self.proximity_mode == "epsilon-ball":
            # Compute density based on the point_mode.
            image_modes = {'triangle', 'square', 'ring', 'star', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
            if self.point_mode == "circle" and not self.density_anomalies:
                if self.dim == 2:
                    density = self.num_points / (np.pi * self.L ** 2)
                elif self.dim == 3:
                    density = self.num_points / (((4 / 3) * np.pi * self.L ** 3))
                else:
                    raise ValueError("Dimension must be 2 or 3.")
            elif self.point_mode in image_modes or (self.point_mode == "circle" and self.density_anomalies):
                density = compute_density(self.positions, self.dim)

            else:
                raise ValueError("Invalid point mode for density computation.")


            # Compute the radius using the helper function.
            radius = compute_epsilon_ball_radius(density, self.intended_av_degree, self.dim, self.proximity_mode)
            distances, indices = compute_epsilon_ball_graph(self.positions, radius)

        elif self.proximity_mode == "delaunay":
            distances, indices = get_delaunay_neighbors(self.positions)

        elif self.proximity_mode == "delaunay_corrected":
            distances, indices = get_delaunay_neighbors_corrected(self.positions)

        elif self.proximity_mode == "distance_decay":
            quantile = self.distance_decay_quantile if self.distance_decay_quantile is not None else 0.02
            distances, indices = compute_proximity_decay_graph(self.positions, quantile_scale=quantile)

        elif self.proximity_mode == "knn_bipartite":
            distances, indices = knn_bipartite(self.positions, self.intended_av_degree)

        elif self.proximity_mode == "epsilon_bipartite":
            # Again, using a placeholder radius (e.g., 0.1) or computing it from density
            radius = 0.1
            distances, indices = epsilon_bipartite(self.positions, radius)

        elif self.proximity_mode == "lattice":
            # For a lattice, we assume the positions are generated by a lattice generator.
            # Use a dedicated lattice computation (e.g., compute_lattice) to determine neighbors.
            distances, indices = compute_lattice(self, self.positions)

        else:
            raise ValueError("Unsupported proximity mode")

        # Build an undirected edge list (avoid duplicates by sorting the node indices)
        edges = set()
        for i, neigh in enumerate(indices):
            for j in neigh:
                edge = tuple(sorted((i, j)))
                edges.add(edge)
        self.edge_list = list(edges)
        return self.edge_list

    def get_positions_df(self):
        """
        Return the positions as a pandas DataFrame with a 'node_ID' column.
        """
        if self.positions is None:
            self.generate_positions()
        if self.config.dim == 2:
            col_names = ['x', 'y']
        elif self.config.dim == 3:
            col_names = ['x', 'y', 'z']
        else:
            raise ValueError("Invalid dimension")
        df = pd.DataFrame(self.positions, columns=col_names)
        df['node_ID'] = range(len(self.positions))
        return df

    def get_edge_list(self, as_dataframe: bool = False):
        """
        Return the computed edge list as either a list of tuples or a pandas DataFrame.
        The DataFrame will contain columns: 'source' and 'target' (and possibly 'distance' if weighted).
        """
        if self.edge_list is None:
            self.compute_graph()
        if as_dataframe:
            # Ensure positions have been generated so we can have node_IDs.
            pos_df = self.get_positions_df()
            df = pd.DataFrame(self.edge_list, columns=['source', 'target'])
            return df
        return self.edge_list

    def get_networkx_graph(self):
        """
        Return a NetworkX Graph built from the edge list.
        Node positions are added as node attributes 'pos'.
        """
        if self.edge_list is None:
            self.compute_graph()
        G = nx.Graph()
        pos_df = self.get_positions_df()
        # Add nodes with position attributes.
        for _, row in pos_df.iterrows():
            G.add_node(row['node_ID'], pos=row.to_dict())
        # Add edges.
        G.add_edges_from(self.edge_list)
        return G