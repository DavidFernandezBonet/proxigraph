# proxigraph/points.py

import numpy as np
from PIL import Image


def generate_random_points(num_points, L, dim):
    """
    Generate random points uniformly distributed in [0, L]^dim.

    Parameters:
        num_points (int): Number of points.
        L (float): Side length of the square or cube.
        dim (int): Dimension (2 or 3).

    Returns:
        List of tuples representing points.
    """
    if dim not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3.")
    points = np.random.rand(num_points, dim) * L
    return [tuple(point) for point in points]


def generate_random_points_in_circle_or_sphere(num_points, R, dim):
    """
    Generate random points uniformly distributed inside a circle (2D) or sphere (3D).

    Parameters:
        num_points (int): Number of points.
        R (float): Radius.
        dim (int): Dimension (2 or 3).

    Returns:
        List of tuples representing points.
    """
    if dim not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3.")
    points = []
    while len(points) < num_points:
        pt = np.random.uniform(-R, R, dim)
        if np.sum(pt ** 2) <= R ** 2:
            points.append(tuple(pt))
    return points


def generate_points_from_image(num_points, image_path):
    """
    Generate points from a black-and-white image.
    Points are generated only where the image is black.

    Parameters:
        num_points (int): Number of points.
        image_path (str): Path to the image file.

    Returns:
        List of tuples representing points.
    """
    with Image.open(image_path) as img:
        img = img.convert('1')  # Convert to black & white
        image_data = np.array(img)
        height, width = image_data.shape

    points = []
    while len(points) < num_points:
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        ix, iy = int(x), int(y)
        if image_data[iy, ix] == 0:
            # Adjust y so that the origin is at the bottom left
            points.append((x, height - y))
    return points


def generate_points_from_image_with_anomalies(num_points, image_path):
    """
    Generate points from a black-and-white image with density anomalies.

    Parameters:
        num_points (int): Number of points.
        image_path (str): Path to the image file.

    Returns:
        List of tuples representing points.
    """
    with Image.open(image_path) as img:
        img = img.convert('1')
        width, height = img.size
        image_data = np.array(img)

    points = set()
    # Define example density regions with multipliers
    density_regions = {
        (0, 0, width / 2, height / 2): 10,
        (width / 2, 0, width / 2, height / 2): 1,
        (0, height / 2, width / 2, height / 2): 5,
        (width / 2, height / 2, width / 2, height / 2): 20
    }
    while len(points) < num_points:
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        density_multiplier = 1
        for (rx, ry, rw, rh), multiplier in density_regions.items():
            if rx <= x < rx + rw and ry <= y < ry + rh:
                density_multiplier = multiplier
                break
        ix, iy = int(x), int(y)
        if image_data[iy, ix] == 0 and np.random.rand() < 1.0 / density_multiplier:
            points.add((x, height - y))
    return list(points)


def generate_square_lattice(args):
    """
    Generate a square lattice of points.

    Parameters:
        args: An object with attributes 'num_points', 'dim', and 'L'.

    Returns:
        A NumPy array of lattice points.
    """
    import numpy as np
    points_per_side = int(np.round(args.num_points ** (1 / args.dim)))
    points = np.linspace(0, args.L, points_per_side)
    mesh = np.meshgrid(*([points] * args.dim))
    return np.array(mesh).T.reshape(-1, args.dim)



