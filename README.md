
# ProxiGraph

ProxiGraph is a Python package for generating proximity graphs (spatial networks) from point clouds.

We provide several methods for generating point clouds and constructing proximity graphs from them! 

![ProxiGraph Examples](proxigraph_examples.png)

## Features

- **Point Generation:**  
  Generate random point clouds using different methods:
  - Shapes: circle, square, star, numbers, custom
  - Dimensions: 2D or 3D (sphere, cube)
  - Densities: Uniform and non-uniform density

- **Graph Construction:**  
  Build proximity graphs using various modes:
  - k-Nearest Neighbors (kNN)
  - Epsilon-ball 
  - Delaunay triangulation (standard and corrected for boundaries)
  - Distance decay graphs
  - Bipartite versions (knn_bipartite, epsilon_bipartite)
  - Lattice graphs

- **Data Export:**  
  Easily retrieve node positions and the edge list as Pandas DataFrames.

- **NetworkX Integration:**  
  Build a NetworkX graph with node positions stored as node attributes.

- **Plotting:**  
  Simple helper functions to visualize your graph using matplotlib.

## Installation

Install ProxiGraph via pip 

```bash
pip install proxigraph
```


## Get-Started Example

Below is an example showing how to generate a graph, retrieve the edges and positions as DataFrames, get a NetworkX graph, and plot the graph:

```python
import matplotlib.pyplot as plt
from proxigraph.config import GraphConfig
from proxigraph.core import ProximityGraph
from proxigraph.plot import plot_graph

# Create a configuration instance.
config = GraphConfig(
    dim=2,
    num_points=1000,
    L=1,
    point_mode="circle",
    proximity_mode="delaunay_corrected",
    density_anomalies=False
)

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

# Get NetworkX graph with node positions as attributes.
G = pg.get_networkx_graph()
print("NetworkX Graph:")
print(G)

# Plot the graph :)
fig, ax = plt.subplots(figsize=(8, 6))
plot_graph(positions, edges=edges, config=config, ax=ax)
plt.show()
```


## Config

All configuration options are encapsulated in the `GraphConfig` dataclass. The key attributes include:

- **dim:** Dimension of the point cloud (2 or 3).
- **num_points:** Number of nodes to generate.
- **L:** The size (or radius) of the domain.
- **point_mode:** Method for generating points (e.g., `"square"`, `"circle"`, or image-based modes like `"triangle"`, `"star"`, etc.).
- **proximity_mode:** Graph construction method (e.g., `"knn"`, `"epsilon-ball"`, `"delaunay"`, `"lattice"`, etc.).
- **intended_av_degree:** Intended average degree for the graph.
- **density_anomalies:** Boolean flag indicating whether to generate density anomalies.

Note: The **Proximity Modes** are briefly described in the [this file](prox_modes.md).


## License

This project is licensed under the MIT License.


