import trimesh
import numpy as np

from src.scrapers.bag3d_scraper import get_surfaces
from vedo.shapes import Mesh
from vedo import merge

from cjio.models import Geometry


def to_vedo_surface(surfaces):
    return [surfaces, [tuple(range(len(surfaces)))]]


def surface_to_vedo_mesh(surfaces, color='red', alpha=0.5):
    meshes = []

    for surface in surfaces:
        mesh = Mesh(to_vedo_surface(surface[0]), c=color, alpha=alpha)
        meshes.append(mesh)
    merged_mesh = merge(meshes)
    return merged_mesh


def geometry_part_to_vedo_mesh(city_object, triangulate=False, lod=2.2):
    """
    Args:
        city_object (_type_): _description_
        lod (float, optional): _description_. Defaults to 2.2.

    Returns:
        (vedo.Mesh, vedo.Mesh, vedo.Mesh): ground, wall, roof
    """
    for geom in city_object.geometry:
        if geom.lod == lod:
            roof_surfaces = get_surfaces(geom, 'RoofSurface')
            wall_surfaces = get_surfaces(geom, 'WallSurface')
            ground_surfaces = get_surfaces(geom, 'GroundSurface')

            roof = surface_to_vedo_mesh(roof_surfaces, color='red', alpha=0.5)
            wall = surface_to_vedo_mesh(wall_surfaces, color='grey', alpha=0.2)
            ground = surface_to_vedo_mesh(ground_surfaces, color='green', alpha=0.7)

            if triangulate:
                roof = roof.triangulate()
                wall = wall.triangulate()
                ground = ground.triangulate()
            return ground, wall, roof
    print(f'LOD {lod} not found')
    return None, None, None


def geometry_part_to_trimesh(city_object, lod=2.2):
    """
    Args:
        city_object (_type_): _description_
        lod (float, optional): _description_. Defaults to 2.2.

    Returns:
        (trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh): ground, wall, roof
    """
    ground, wall, roof = geometry_part_to_vedo_mesh(city_object, triangulate=True, lod=lod)

    if ground and wall and roof:
        roof = trimesh.Trimesh(roof.points(), roof.faces())
        wall = trimesh.Trimesh(wall.points(), wall.faces())
        ground = trimesh.Trimesh(ground.points(), ground.faces())

        return ground, wall, roof
    
    else:
        return None, None, None


def trimesh_to_geometry(mesh, lod=5.0):
    """
    Args:
        mesh (trimesh.Trimesh): Building model

    Returns:
        cjio.models.Geometry: Building model
    """
    geom = Geometry(type='Solid', lod=lod)

    # Obtain boundaries
    boundaries = mesh.triangles.reshape(-1, 1, 3, 3).tolist()
    geom.boundaries.append(boundaries)

    # Obtain surfaces (roof and wall only) from colors
    C = mesh.visual.face_colors # All colors
    r = np.array([255, 0, 0]) # roof color
    mask = np.where(((C[:,0] == r[0]) & (C[:,1] == r[1]) & (C[:,2] == r[2])), True, False)
    space = np.arange(mesh.faces.shape[0])

    # Add surface info to geometry
    geom.surfaces[0] = {'surface_idx': np.vstack((np.zeros_like(space[np.logical_not(mask)]), space[np.logical_not(mask)])).T.tolist(), 'type': 'WallSurface'}
    geom.surfaces[1] = {'surface_idx': np.vstack((np.zeros_like(space[mask]), space[mask])).T.tolist(), 'type': 'RoofSurface'}
    return geom