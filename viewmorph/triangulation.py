from scipy.spatial import Delaunay


def get_triangles(points):
    return Delaunay(points)
