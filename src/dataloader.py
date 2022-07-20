import numpy as np
import trimesh
import os
import laspy
import networkx as nx

from src.utils.cityjson import geometry_part_to_trimesh


def get_floorplan_from_mesh(mesh):
    G = nx.Graph()

    for a, b in mesh.edges:
        ca, cb = mesh.vertices[a,2], mesh.vertices[b,2]
        if ca < 0.01 and cb < 0.01:
            G.add_edge(tuple(mesh.vertices[a,:2]), tuple(mesh.vertices[b,:2]))
    floorplan = nx.cycle_basis(G)

    # Make round
    floorplan = [part + [part[0]] for part in floorplan]
    # To list
    return [[list(node) for node in part] for part in floorplan]


def get_item(id, dataset_root, city_model=None, city_map=None, city_outline=None, return_aer=False, center=True):
    """_summary_

    Args:
        id (_type_): _description_
        city_model (_type_, optional): _description_. Defaults to None.
        city_map (_type_, optional): _description_. Defaults to None.
        dataset (str, optional): _description_. Defaults to 'Amsterdam'.

    Returns:
        trimesh.Trimesh, trimesh.Trimesh, list[list[list]], shapely.multipolygon, laspy : wall, roof, floorplan, outline, pcd
    """
    assert city_model, 'city_model required for Amsterdam dataset'
    assert city_map, 'city_map required for Amsterdam dataset'
    assert city_outline, 'city_outline required for Amsterdam dataset'

    pcd_aer_dir = 'D:/datasets/Amsterdam/aerial/'

    # Load building data
    building = city_model.get_cityobjects(type=['building', 'buildingpart'])[f'NL.IMBAG.Pand.{id}-0']
    _, wall, roof = geometry_part_to_trimesh(building)
    floorplan = city_map[id]['floorplan']
    outline = city_outline[id]['outline']
    pcd = laspy.read(os.path.join(dataset_root, id + '.laz'))

    if return_aer:
        pcd_aer = laspy.read(os.path.join(pcd_aer_dir, id + '.laz'))

        pcd_aer.red *= 256
        pcd_aer.green *= 256
        pcd_aer.blue *= 256


    if center:
        mean = np.array([np.array(pcd.x).mean(), np.array(pcd.y).mean(), np.array(pcd.z).mean()])
        wall.vertices -= mean
        roof.vertices -= mean
        pcd.x -= mean[0]
        pcd.y -= mean[1]
        pcd.z -= mean[2]

        floorplan = [[[c0 - mean[0], c1 - mean[1]] for c0, c1 in part] for part in floorplan]
        outline = [[[c0 - mean[0], c1 - mean[1]] for c0, c1 in part] for part in outline]
    
    upscale = False
    if upscale:
        pcd.x *= 2
        pcd.y *= 2
        pcd.z *= 2

    return wall, roof, floorplan, outline, pcd