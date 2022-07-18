import trimesh
import numpy as np
import os
import trimesh
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from timeit import default_timer as timer

from src.utils.meshes import merge_wall_and_floorplan3d, add_color_to_mesh, floorplan3dfier, get_intersection
from src.utils.metrics import average_nearest_neighbour
import src.dataloader as dataloader


def sample_even_pcd(pcd, N=10000, bins=20, bottom=0.0):
    pcd = pcd[np.where(pcd.z > bottom)]
    clip_idx = []

    space, stepsize = np.linspace(pcd.z.min(), pcd.z.max(), bins, retstep=True)
    num_bins = len(space)

    for bin in space: 
        bin_idx = np.where((pcd.z > bin) & (pcd.z <= bin + stepsize))[0]
        sample_idx = np.zeros(len(pcd))
        if bin_idx.shape[0] >= N / num_bins:
            sample_size = int(N / num_bins)
        else:
            sample_size = bin_idx.shape[0]

        sample_idx = bin_idx[np.random.choice(bin_idx.shape[0], sample_size, replace=False)]
        clip_idx += list(sample_idx)

    return pcd[clip_idx]


def get_samples(building_model, floorplan3d, intersection, pcd, N=10000, bottom_buffer=2.0):
    """_summary_

    Args:
        building_model (trimesh.Trimesh): _description_
        floorplan3d (trimesh.Trimesh): _description_
        intersection (trimesh.Trimesh): _description_
        pointcloud (laspy): _description_
        N (int, optional): _description_. Defaults to 10000.
        margin (float, optional): _description_. Defaults to 2.0.

    Returns:
        samples_building_model (np.array(N,3)),
        samples_floorplan3d (np.array(N,3)),
        samples_intersection (np.array(N,3)),
        samples_pointcloud (np.array(N,3)),
    """

    if intersection:
        total_area = sum([building_model.area, floorplan3d.area, intersection.area])

        samples_building_model, _ = trimesh.sample.sample_surface_even(building_model, int(N * (building_model.area / np.   sum([building_model.area, intersection.area]))))
        samples_floorplan3d, _ = trimesh.sample.sample_surface_even(floorplan3d, int(N * (floorplan3d.area / np.sum([floorplan3d.area, intersection.area]))))
        samples_intersection, _ = trimesh.sample.sample_surface_even(intersection, int(N * (intersection.area / np.sum([np.mean([floorplan3d.area, building_model.area]), intersection.area]))))

    else:
        total_area = sum([building_model.area, floorplan3d.area])

        samples_building_model, _ = trimesh.sample.sample_surface_even(building_model, int(N * (building_model.area / total_area)))
        samples_floorplan3d, _ = trimesh.sample.sample_surface_even(floorplan3d, int(N * (floorplan3d.area / total_area)))
        samples_intersection = None

    pcd = pcd[np.where(pcd.z > building_model.vertices.min(axis=0)[2] + bottom_buffer)]
    pcd = pcd[np.random.choice(len(pcd), min(N, len(pcd)), replace=False)]

    samples_pointcloud = np.array([pcd.x, pcd.y, pcd.z]).T
    return samples_building_model, samples_floorplan3d, samples_intersection, samples_pointcloud


def optimal_intersection_height(samples_building_model, samples_floorplan3d, samples_intersection, samples_pointcloud, stepsize=0.2, smooth=False):
    """_summary_

    Args:
        samples_building_model (np.array(N,3)): _description_
        samples_floorplan3d (np.array(N,3)): _description_
        samples_intersection (np.array(N,3)): _description_
        samples_building (np.array(N,3)): _description_
        stepsize (float, optional): _description_. Defaults to 0.2.
        buffer (float, optional): _description_. Defaults to 3.0.
        smooth (bool, optional): _description_. Defaults to False.

    Returns:
        height: (float),
        score: (float),
        scores: (np.array(M,)),
        steps: (np.array(M,))
    """
    # mean height from the building model ground
    min_height = (samples_building_model.min(axis=0)[2] // stepsize) * stepsize
    max_height = (samples_building_model.max(axis=0)[2] // stepsize) * stepsize

    steps = np.arange(min_height, max_height, stepsize)
    scores = np.zeros_like(steps)

    # Exit if building is too small
    if steps.shape[0] == 0:
        return None, np.inf, None, None

    samples_floorplan_z = samples_floorplan3d[:,2]
    samples_building_model_z = samples_building_model[:,2]

    for i, intersection in enumerate(steps):
        # Generate a cloud of the new merged building
        samples_building = np.vstack((
            samples_floorplan3d[np.where(samples_floorplan_z < intersection, True, False)], 
            samples_building_model[np.where(samples_building_model_z >= intersection, True, False)],
            ))
        if not samples_intersection is None:
            samples_building = np.vstack((
                samples_building,
                samples_intersection + np.array([0.0, 0.0, intersection])
            ))

        # Construct a compact tree
        tree_pointcloud = cKDTree(samples_building, compact_nodes=False, balanced_tree=False)

        distances, _  = tree_pointcloud.query(samples_pointcloud, workers=-1)
        ann = distances.mean()
        scores[i] = ann

    if smooth:
        # Apply a gaussian filter
        scores = np.convolve(scores, np.array([0.1, 0.2, 0.4, 0.2, 0.1]), mode='valid')
        steps = steps[2:-2]

    # Compute the extrema
    height = steps[scores.argmin()]
    score = scores.min()

    return height, score, scores, steps


def intersect(out_folder, idx, dataset='Amsterdam', stepsize=0.1, N=10000, improvement_threshold=0.0, bottom_buffer=3.0, smooth=True, city_model=None, city_map=None, city_outline=None):
    """_summary_

    Args:
        out_folder (str): 
        idx (list): List of strings containing building idx for the dataloader
        dataset (str, optional): Switch for different datasets. Defaults to 'Amsterdam'.
        stepsize (float, optional): Defines the interval to check the intersection. Defaults to 0.1.
        N (int, optional): Number of samples. Defaults to 10000.
        improvement_threshold (float, optional): Threshold to determine when the improvement is good enough. Defaults to 0.0.
        bottom_buffer (float, optional): Defines the relative lowest intersection height. Defaults to 3.0.
        buffer (float, optional): _description_. Defaults to 3.0.
        city_model (cjio.cityjson.CityJSON, optional): City model Defaults to None.
        city_map (list[list[list[]]], optional): lists containing building floorplans. Defaults to None.
        city_outline (list[list[list[]]], optional): lists containing building outlines. Defaults to None.

    Returns:
        _type_: _description_
    """

    results = []

    for i, id in enumerate(idx):
        # Load building data
        wall, roof, floorplan, outline, pcd = dataloader.get_item(id, city_model=city_model, city_map=city_map, city_outline=city_outline, dataset=dataset, return_aer=True, center=False)

        # Create a 3D version of the footprint
        floorplan3d = floorplan3dfier(floorplan, bottom_plane=False, top_plane=False, bottom=wall.vertices.min(axis=0)[2], top=wall.vertices.max(axis=0)[2])

        # Create a 3D version of the intersection

        try:
            intersection = get_intersection(floorplan, outline, 0)
        except:
            intersection = None

        # Sample points
        samples_wall, samples_floorplan3d, samples_intersection, samples_pointcloud = get_samples(trimesh.util.concatenate(wall, roof), floorplan3d, intersection, pcd, N=N, bottom_buffer=bottom_buffer)
        
        # Compute the original ann score
        building_ann_score = average_nearest_neighbour(samples_pointcloud, wall, N=N)

        # Compute the optimal intersection height
        start = timer()
        optimal_height, intersected_ann_score, _, _ = optimal_intersection_height(samples_wall, samples_floorplan3d, samples_intersection, samples_pointcloud, stepsize=stepsize, smooth=smooth)
        time = timer() - start

        # If improvement is bigger than a threshold, export new building
        if (building_ann_score - intersected_ann_score) > improvement_threshold:
            floorplan3d = floorplan3dfier(floorplan, bottom_plane=False, top_plane=False, bottom=wall.vertices.min(axis=0)[2], top=optimal_height)
            output_wall = merge_wall_and_floorplan3d(wall, floorplan3d, intersection_height=optimal_height)
            intersection = get_intersection(floorplan, outline, optimal_height)
        
        # Otherwise, return original building
        else:
            optimal_height = None
            output_wall = wall
            intersected_ann_score = building_ann_score

        # Add colors
        add_color_to_mesh(output_wall, [1.0, 1.0, 1.0])
        add_color_to_mesh(roof, [1.0, 0.0, 0.0])

        if optimal_height and intersection:
                add_color_to_mesh(intersection, [1.0, 1.0, 0.0])
                output_wall = trimesh.util.concatenate([output_wall, intersection])
        full_building = trimesh.util.concatenate([output_wall, roof])

        # Write new mesh
        trimesh.exchange.export.export_mesh(full_building, os.path.join(out_folder, id + '.obj'))

        # Log results
        result = [id, intersected_ann_score, len(full_building.facets), full_building.faces.shape[0], time]
        results.append(result)
        print(i, result, optimal_height)

    results_summary = pd.DataFrame(results, columns = ['id', 'intersected_ann_score', 'intersected_facets', 'intersected_triangles', 'time']).set_index('id')
    return results_summary