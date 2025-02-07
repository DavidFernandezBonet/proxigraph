# proxigraph/config.py

class GraphConfig:
    def __init__(self,
                 dim=2,
                 num_points=1000,
                 L=1.0,
                 point_mode="circle",
                 proximity_mode="delaunay_corrected",
                 intended_av_degree=10,
                 distance_decay_quantile=None,
                 density_anomalies=False,
                 show_plots=True,
                 directory_map=None,
                 color_mapping=None):
        """
        Parameters:
          dim (int): Dimension of the data (2 or 3).
          num_points (int): Number of nodes.
          L (float): Size of the domain.
          point_mode (str): Mode for point generation (e.g., "square", "circle", etc.).
          proximity_mode (str): Graph construction mode (e.g., "knn", "epsilon-ball", "delaunay", etc.).
          intended_av_degree (int): Target average degree.
          distance_decay_quantile (float): Scaling quantile for decay graphs.
          density_anomalies (bool): Whether to generate density anomalies.
          show_plots (bool): Whether to display plots.
          directory_map (dict): Dictionary of directory paths.
          color_mapping (dict): Optional mapping (node_ID → color) to override default colors.
        """
        self.dim = dim
        self.num_points = num_points
        self.L = L
        self.point_mode = point_mode
        self.proximity_mode = proximity_mode
        self.intended_av_degree = intended_av_degree
        self.distance_decay_quantile = distance_decay_quantile
        self.density_anomalies = density_anomalies
        self.show_plots = show_plots
        self.directory_map = directory_map or {}
        self.color_mapping = color_mapping
