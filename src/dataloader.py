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


def get_item(id, city_model=None, city_map=None, city_outline=None, return_aer=False, dataset='Amsterdam', center=True):
    """_summary_

    Args:
        id (_type_): _description_
        city_model (_type_, optional): _description_. Defaults to None.
        city_map (_type_, optional): _description_. Defaults to None.
        dataset (str, optional): _description_. Defaults to 'Amsterdam'.

    Returns:
        trimesh.Trimesh, trimesh.Trimesh, list[list[list]], shapely.multipolygon, laspy : wall, roof, floorplan, outline, pcd
    """
    if dataset == 'Toy':
        like_bag_root = 'C:/Users/boble/Documents/AI-year2/Thesis/data/evaluation/toy_like_bag'
        pcd_root = 'C:/Users/boble/Documents/AI-year2/Thesis/data/evaluation/clouds'
        # pcd_root = 'D:/datasets/Toy/cloud_mf_1'
        gt_root = 'C:/Users/boble/Documents/AI-year2/Thesis/data/evaluation/references'

        # Load floorplan
        gt_mesh = trimesh.exchange.load.load(os.path.join(gt_root, id + '.obj'), force='mesh')
        floorplan = get_floorplan_from_mesh(gt_mesh)

        # Load roof and wall
        building = trimesh.exchange.load.load(os.path.join(like_bag_root, id + '.obj'))
        for t in building.geometry.values():
            if np.array_equal(t.visual.material.main_color, np.array([255, 255, 255, 255])):
                wall = t
            elif np.array_equal(t.visual.material.main_color, np.array([255, 0, 0, 255])):
                roof = t

        # Load outline
        building25 = trimesh.util.concatenate(wall, roof)
        outline = trimesh.path.polygons.projected(building25, normal=[0,0,1])
        outline = np.round(np.array(outline.exterior.xy).T, 3).reshape(1,-1, 2).tolist()

        # Load pcd
        pcd = laspy.read(os.path.join(pcd_root, id + '.laz'))

    elif dataset == 'BuildingNet':
        gt_root = 'D:/datasets/BuildingNet/gt'
        pcd_root = 'D:/datasets/BuildingNet/cloud'

        # Load pcd
        pcd = laspy.read(os.path.join(pcd_root, id + '.laz'))

        # Load floorplan
        gt_mesh = trimesh.exchange.load.load(os.path.join(gt_root, id + '.obj'), force='mesh')
        floorplan = get_floorplan_from_mesh(gt_mesh)
        wall = None
        roof = None

        # Compute outline
        try:
            outline = trimesh.path.polygons.projected(gt_mesh, normal=[0,0,1])
            outline = np.round(np.array(outline.exterior.xy).T, 3).reshape(1,-1, 2).tolist()
        except:
            print('outline failed')
            outline = np.array([[[]]])


    elif dataset == 'Amsterdam':
        assert city_model, 'city_model required for Amsterdam dataset'
        assert city_map, 'city_map required for Amsterdam dataset'
        assert city_outline, 'city_outline required for Amsterdam dataset'

        pcd_dir = 'D:/datasets/Amsterdam/filtered/'
        pcd_aer_dir = 'D:/datasets/Amsterdam/aerial/'

        # Load building data
        building = city_model.get_cityobjects(type=['building', 'buildingpart'])[f'NL.IMBAG.Pand.{id}-0']
        _, wall, roof = geometry_part_to_trimesh(building)
        floorplan = city_map[id]['floorplan']
        outline = city_outline[id]['outline']
        pcd = laspy.read(os.path.join(pcd_dir, id + '.laz'))

        if return_aer:
            pcd_aer = laspy.read(os.path.join(pcd_aer_dir, id + '.laz'))

            pcd_aer.red *= 256
            pcd_aer.green *= 256
            pcd_aer.blue *= 256

    else:
        print(f'Type {type} unknown')
        # return None, None, None

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