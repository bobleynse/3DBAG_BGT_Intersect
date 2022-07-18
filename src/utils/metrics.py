import trimesh
import numpy as np
from scipy.spatial import cKDTree


def get_distances(cloud_a, cloud_b):
    kd_tree = cKDTree(cloud_b, compact_nodes=False, balanced_tree=False)
    distances, _ = kd_tree.query(cloud_a, k=1, workers=-1)
    
    return distances


def average_nearest_neighbour(pcd, trmesh, N=10000):

    pcd_samples = pcd[np.random.choice(len(pcd), min(N, len(pcd)), replace=False)]
    mesh_samples, _ = trimesh.sample.sample_surface_even(trmesh, N)
    distances = get_distances(pcd_samples, mesh_samples)
    
    return distances.mean()


def average_nearest_neighbour_ratio(pcd, trmesh, A=None, N=10000):
    ann = average_nearest_neighbour(pcd, trmesh, N)

    min = trmesh.vertices.min(axis=0)
    max = trmesh.vertices.max(axis=0)

    if not A:
        A = np.product(np.abs(min - max))

    return ann / (0.5 / np.sqrt(N / A))