import trimesh
import numpy as np

from shapely.geometry import Polygon, MultiPolygon
from vedo.mesh import Mesh
from vedo import merge


def to_vedo_surface(surfaces):
    return [surfaces, [tuple(range(len(surfaces)))]]


def floorplan3dfier_vedo(floorplan, bottom=0, top=3, bottom_plane=False, top_plane=False, color='green', alpha=0.7, triangulate=False):
    """Constructs a 3D version of a 2d floorplan

    Args:
        floorplan (list[list[list]]]): list of polygons
        bottom (int, optional): Bottom height of the new shape. Defaults to 0.
        top (int, optional): Top height of the new shape. Defaults to 3.
        bottom_plane (bool, optional): Add a bottom polygon?. Defaults to False.
        top_plane (bool, optional): Add a bottom polygon?. Defaults to False.
        color (str, optional): Change color. Defaults to 'green'.
        alpha (float, optional): Add opacity. Defaults to 0.7.
        triangulate (bool, optional): Convert polygons to triangles. Defaults to False.

    Returns:
        vedo.Mesh
    """

    meshes = []

    for polygon in floorplan:
        for point_a, point_b in zip(polygon[:-1], polygon[1:]):
            poly = [point_a + [bottom], point_a + [top], point_b + [top], point_b + [bottom]]
            mesh = Mesh([poly, [[0,1,2,3]]], c=color, alpha=alpha)
            meshes.append(mesh)

        if bottom_plane:
            poly = [point + [bottom] for point in polygon]
            mesh = Mesh(to_vedo_surface(poly), c='blue', alpha=alpha)
            meshes.append(mesh)

        if top_plane:
            poly = [point + [top] for point in polygon]
            mesh = Mesh(to_vedo_surface(poly), c='blue', alpha=alpha)
            meshes.append(mesh)

    meshes = merge(meshes)
    if triangulate:
        meshes = meshes.triangulate()

    return meshes


def floorplan3dfier(floorplan, bottom=0, top=3, bottom_plane=False, top_plane=False):
    """Constructs a 3D version of a 2d floorplan

    Args:
        floorplan (list[list[list]]]): list of polygons
        bottom (int, optional): Bottom height of the new shape. Defaults to 0.
        top (int, optional): Top height of the new shape. Defaults to 3.
        bottom_plane (bool, optional): Add a bottom polygon?. Defaults to False.
        top_plane (bool, optional): Add a bottom polygon?. Defaults to False.

    Returns:
        trimesh.Trimesh
    """
    vedo_mesh = floorplan3dfier_vedo(floorplan, bottom=bottom, top=top, bottom_plane=bottom_plane, top_plane=top_plane, triangulate=True)
    return trimesh.Trimesh(vedo_mesh.points(), vedo_mesh.faces())


def get_intersection(floorplan, outline, height=5):
    """Compute the intersection mesh between the 3Dfied floorplan and 2.5D building model.

    Args:
        floorplan (list[list[list]]]): Building floorplan
        outline (list[list[list]]]): Building outline
        height (int, optional): Height of the intersection plane. Defaults to 5.

    Returns:
        trimesh.Trimesh
    """
    # Obtain the shapely (sh) floorplan 
    floorplan_polygons = []
    for floorplan_part in floorplan:
        if len(floorplan_part) >= 3:
            floorplan_polygons.append(Polygon(floorplan_part))
    floorplan_sh = MultiPolygon(floorplan_polygons)

    # Obtain the shapely (sh) outline
    outline_polygons = []
    for outline_part in outline:
        if len(outline_part) >= 3:
            outline_polygons.append(Polygon(outline_part))
    outline_sh = MultiPolygon(outline_polygons)    

    # Return the difference
    intersection_up = outline_sh.buffer(0).difference(floorplan_sh.buffer(0))
    intersection_down = floorplan_sh.buffer(0).difference(outline_sh.buffer(0))

    if type(intersection_up) == Polygon:
        intersection_up = [intersection_up]
    if type(intersection_down) == Polygon:
        intersection_down = [intersection_down]

    # Convert to trimesh UP
    intersections_up = []
    for intersection_part in intersection_up:
        try:
            vertices, faces = trimesh.creation.triangulate_polygon(intersection_part)
            vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1)) * height))
            intersections_up.append(trimesh.Trimesh(vertices, faces))
        except:
            print('ERROR: in shapely intersection down')

    if len(intersections_up) > 0:
        up = trimesh.util.concatenate(intersections_up)
    elif len(intersections_up) == 0:
        up = intersections_up
    else:
        up = None

    # Convert to trimesh DOWN
    intersections_down = []
    for intersection_part in intersection_down:
        try:
            vertices, faces = trimesh.creation.triangulate_polygon(intersection_part)
            vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1)) * height))
            intersections_down.append(trimesh.Trimesh(vertices, faces))
        except:
            print('ERROR: in shapely intersection down')
    if len(intersections_down) > 0:
        down = trimesh.util.concatenate(intersections_down)
    elif len(intersections_down) == 0:
        down = intersections_down
    else:
        down = None

    if up and down:
        return trimesh.util.concatenate(up, down)
    elif up and not down:
        return up
    elif not up and down:
        return down


def auto_crop_mesh_bottom(mesh, intersection_height=3.0):
    """Crops the bottom from a mesh. First try the fast trimesh method.
       Otherwise, use the slower trimesh blender option.

    Args:
        mesh (trimesh.Trimesh)
        intersection_height (float, optional): Intersection height. Defaults to 3.0.

    Returns:
        trimesh.Trimesh
    """

    try:
        return trimesh.intersections.slice_mesh_plane(mesh, [0,0,1], [0,0,intersection_height], cap=False)
    except:
        print('WARNING: Fast trimesh cut_mesh_bottom failed')

    maximum_height = 500

    x_min, y_min, _ = mesh.vertices.min(axis=0) - 5
    x_max, y_max, _ = mesh.vertices.max(axis=0) + 5

    roi_mesh = trimesh.Trimesh(
        vertices=np.array([
            [x_min, y_min, intersection_height],
            [x_min, y_max, intersection_height],
            [x_max, y_max, intersection_height],
            [x_max, y_min, intersection_height],
            [x_min, y_min, maximum_height],
            [x_min, y_max, maximum_height],
            [x_max, y_max, maximum_height],
            [x_max, y_min, maximum_height]
            ]),
        faces=np.array([
            [0, 1, 2, 3],
            [7, 6, 5, 4],
            [0, 1, 7, 6],
            [1, 2, 6, 5],
            [2, 3, 5, 4],
            [3, 0, 4, 7]
        ]))
    roi_mesh.fix_normals()
    return trimesh.boolean.intersection([mesh, roi_mesh], 'blender')


def merge_wall_and_floorplan3d(building, floorplan3d, intersection_height=4.0):
    cropped_building = auto_crop_mesh_bottom(building, intersection_height)
    new_mesh = trimesh.util.concatenate(cropped_building, floorplan3d)
    return new_mesh


def add_color_to_mesh(mesh, color):
    """
    Args:
        mesh (trimesh.Trimesh): A mesh
        color ([r,g,b]): RGB color 0-255

    Returns:
        trimesh.Trimesh
    """
    if mesh:
        facet_colors = np.tile(color, (mesh.faces.shape[0], 1))
        facet_colors = trimesh.visual.ColorVisuals(mesh, face_colors=facet_colors)
        mesh.visual = facet_colors
    return mesh