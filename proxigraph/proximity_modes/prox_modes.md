## Proximity Modes

ProxiGraph supports a variety of proximity modes for constructing graphs from point clouds. Below is a brief explanation of each mode:

- **knn:**  
  Connects each node to its *k* nearest neighbors. The parameter `intended_av_degree` determines the number of neighbors (k). This is useful when you want a graph based on local proximity.

- **epsilon-ball:**  
  Connects nodes that are within a specified radius. The radius is computed based on the density of the point cloud. For example, for a uniform circular distribution (with point mode "circle"), the density is computed as the number of points divided by the area of the circle (πL² in 2D). For a square, it uses L² (or L³ in 3D). For image-based or density anomaly cases, the density is computed from the bounding box of the generated positions.

- **delaunay:**  
  Uses Delaunay triangulation to determine neighbor relationships. In this mode, nodes are connected if their corresponding Voronoi regions share a boundary. This mode often produces a sparse, well-connected graph.

- **delaunay_corrected:**  
  Similar to the standard Delaunay mode, but with a correction step. Long edges (typically above a given percentile threshold) are filtered out, and additional edges may be added to ensure the graph remains connected.

- **distance_decay:**  
  Constructs a graph where the probability of forming an edge decays with distance. The decay function can be either exponential or follow a power law. A scaling factor is determined by a specified quantile of the distance distribution, which helps in adjusting the decay to the scale of the data.

- **knn_bipartite:**  
  Divides the node set into two groups (according to a given ratio) and then uses a k-nearest neighbors search across the two groups. This mode builds a bipartite graph where edges connect nodes from different subsets.

- **epsilon_bipartite:**  
  Similar to the epsilon-ball mode but applied in a bipartite context. The node set is partitioned into two groups, and nodes are connected if they fall within the computed radius (using the same density-based calculation) across the groups.

- **lattice:**  
  Assumes that the nodes are generated on a regular grid (lattice). Neighbors are determined based on the regular structure of the grid (e.g., each node might be connected to its four immediate neighbors in 2D, or six in 3D). This mode is useful when your point cloud is naturally arranged in a grid pattern.

You can select a proximity mode by setting the `proximity_mode` attribute in your `GraphConfig` object. For example, to use the epsilon-ball mode with density-based radius computation, you would set `proximity_mode="epsilon-ball"`. Each mode has its own characteristics and is suited for different types of data and applications.
