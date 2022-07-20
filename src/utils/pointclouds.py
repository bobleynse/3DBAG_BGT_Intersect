from shapely.geometry import Polygon
from numba import jit, njit
import numba
import numpy as np
import os
import open3d as o3d
import laspy


def get_bbox_from_tile_code(tile_code, padding=0, width=50, height=50):
    """
    Get the <X,Y> bounding box for a given tile code. The tile code is assumed
    to represent the lower left corner of the tile.

    Parameters
    ----------
    tile_code : str
        The tile code, e.g. 2386_9702.
    padding : float
        Optional padding (in m) by which the bounding box will be extended.
    width : int (default: 50)
        The width of the tile.
    height : int (default: 50)
        The height of the tile.

    Returns
    -------
    tuple of tuples
        Bounding box with inverted y-axis: ((x_min, y_max), (x_max, y_min))
    """
    tile_split = tile_code.split('_')

    # The tile code of each tile is defined as
    # 'X-coordinaat/50'_'Y-coordinaat/50'
    x_min = int(tile_split[0]) * 50
    y_min = int(tile_split[1]) * 50

    return ((x_min - padding, y_min + height + padding),
            (x_min + height + padding, y_min - padding))


def get_bbox_from_tile_codes(tile_codes, padding=0.0):
    """Get the <X,Y> bounding box for a list of tile codes.
    The tile code is assumed to represent the lower left corner of the tile.
    All space in between the tiles will also be included

    Args:
        tile_codes (list of str): The tile codes, e.g. [2386_9702, 2446_9521].
        padding (float): Optional padding (in m) by which the bounding box will be extended.

    Returns:
        tuple of tuples: Bounding box with inverted y-axis: ((x_min, y_max), (x_max, y_min))
    """
    extrema = np.zeros((len(tile_codes), 4))
    
    for i, tile_code in enumerate(tile_codes):
        ((x_min, y_max), (x_max, y_min)) = get_bbox_from_tile_code(tile_code, padding=padding)
        extrema[i] = [x_min, y_max, x_max, y_min]
    return ((extrema[:,0].min(), extrema[:,1].max()), (extrema[:,2].max(), extrema[:,3].min()))


@jit(nopython=True)
def point_in_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in numba.prange(n + 1):
        p2x,p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


@njit(parallel=True)
def numba_parallel_points_in_polygon(points, polygon):
    D = np.empty(len(points), dtype=numba.boolean) 
    for i in numba.prange(0, len(D)):
        D[i] = point_in_polygon(points[i,0], points[i,1], polygon)
    return D


def filter_roi(pcd, buffer_inside=0.0, buffer_outside=0.0, bgt_floorplan=None, bag_floorplan=None):
    if bgt_floorplan:
        clip_idx_inside_floorplan = np.zeros(len(pcd), dtype=bool)
        # Remove every point within a buffer inside of the floorplan
        if buffer_inside != 0:
            for polygon in bgt_floorplan:
                clip_idx = numba_parallel_points_in_polygon(np.array([pcd.x, pcd.y]).T, np.array(Polygon(polygon).buffer(buffer_inside, cap_style=2, resolution=2).exterior))
                clip_idx_inside_floorplan = np.logical_or(clip_idx_inside_floorplan, clip_idx)
            pcd = pcd[np.invert(clip_idx_inside_floorplan)]

        # # Keep every point within a buffer outside of the floorplan
        # clip_idx_outside_floorplan = np.zeros(len(pcd), dtype=bool)

        # if buffer_outside != 0:
        #     for polygon in bgt_floorplan:
        #         clip_idx = numba_parallel_points_in_polygon(np.array([pcd.x, pcd.y]).T, np.array(Polygon(polygon).buffer(buffer_outside, cap_style=2, resolution=2).exterior))
        #         clip_idx_outside_floorplan = np.logical_or(clip_idx_outside_floorplan, clip_idx)
        #     pcd = pcd[clip_idx_outside_floorplan]

    if bag_floorplan:
        # clip_idx_inside_floorplan = np.zeros(len(pcd), dtype=bool)
        # # Remove every point within a buffer inside of the floorplan
        # if buffer_inside != 0:
        #     for polygon in bag_floorplan:
        #         clip_idx = numba_parallel_points_in_polygon(np.array([pcd.x, pcd.y]).T, np.array(Polygon(polygon).buffer(buffer_inside, cap_style=2, resolution=2).exterior))
        #         clip_idx_inside_floorplan = np.logical_or(clip_idx_inside_floorplan, clip_idx)
        #     pcd = pcd[np.invert(clip_idx_inside_floorplan)]

        # Keep every point within a buffer outside of the floorplan
        clip_idx_outside_floorplan = np.zeros(len(pcd), dtype=bool)

        if buffer_outside != 0:
            for polygon in bag_floorplan:
                clip_idx = numba_parallel_points_in_polygon(np.array([pcd.x, pcd.y]).T, np.array(Polygon(polygon).buffer(buffer_outside, cap_style=2, resolution=2).exterior))
                clip_idx_outside_floorplan = np.logical_or(clip_idx_outside_floorplan, clip_idx)
            pcd = pcd[clip_idx_outside_floorplan]

    return pcd


def clean_pointcloud_tiles(input_dir, output_dir, save_colors=True, voxel_size=0.1):
    for file in os.listdir(input_dir):
        print('Processing:', file)
        lascloud = laspy.read(os.path.join(input_dir, file))

        xyz = np.array([lascloud.x, lascloud.y, lascloud.z]).T
        rgb = np.array([lascloud.red, lascloud.green, lascloud.blue]).T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Reduce size
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        output_lascloud = laspy.create()
        output_lascloud.x = np.array(pcd.points)[:,0]
        output_lascloud.y = np.array(pcd.points)[:,1]
        output_lascloud.z = np.array(pcd.points)[:,2]

        if save_colors:
            output_lascloud.red = np.array(pcd.colors)[:,0]
            output_lascloud.green = np.array(pcd.colors)[:,1]
            output_lascloud.blue = np.array(pcd.colors)[:,2]

        output_lascloud.write(os.path.join(output_dir, 'cleaned_' + file))      
        


def divide_tiles_per_building(input_dir, output_dir, city_map, city_outlines, buffer_inside=0.5, buffer_outside=0.5, save_colours=True):
    for id, outline in list(city_outlines.items()):
        print('Processing:', id)

        # Get all points in floorplan
        coords_list = []
        for polygon in outline['outline']:
            for coord in polygon:
                coords_list.append(coord)
        coords = np.array(coords_list)

        # # Get all rijksdriehoek extrema of the polygon
        # rd_x_min, rd_y_min = coords.min(axis=0) - buffer_outside
        # rd_x_max, rd_y_max = coords.max(axis=0) + buffer_outside

        # Get all cyclomedia floor extrema of the polygon
        cm_x_min, cm_y_min = np.floor((coords.min(axis=0) - buffer_outside) / 50).astype(int) 
        cm_x_max, cm_y_max = np.floor((coords.max(axis=0) + buffer_outside) / 50).astype(int) 

        # Check if all required pointcloud tiles are available
        id_filedirs = []
        id_filedirs_available = []

        for rd_x in np.arange(cm_x_min, cm_x_max + 1, 1):
            for rd_y in np.arange(cm_y_min, cm_y_max + 1, 1):
                filedir = os.path.join(input_dir, f'cleaned_filtered_{rd_x}_{rd_y}.laz')
                id_filedirs.append(filedir)
                id_filedirs_available.append(os.path.isfile(filedir))

        # When all needed pointclouds available, add info to dict
        if np.array(id_filedirs_available).all():
            # # Loop over all dirs and stack afterwards
            # pcds_to_stack = []
            # N = 0

            x = np.array([])
            y = np.array([])
            z = np.array([])

            if save_colours:
                red = np.array([])
                green = np.array([])
                blue = np.array([])

            for i, dir in enumerate(id_filedirs):
                pcd = laspy.read(dir)

                try:
                    pcd = filter_roi(pcd, buffer_inside, buffer_outside, bgt_floorplan=city_map[id]['floorplan'], bag_floorplan=outline['outline'])
                except:
                    print('WARNING: interior failed')
                    pcd = filter_roi(pcd, 0, buffer_outside, bag_floorplan=outline['outline'])

                x = np.concatenate([x, pcd.x])
                y = np.concatenate([y, pcd.y])
                z = np.concatenate([z, pcd.z])

                if save_colours:
                    red = np.concatenate([red, pcd.red])
                    green = np.concatenate([green, pcd.green])
                    blue = np.concatenate([blue, pcd.blue])

                print(x.shape, y.shape)

            output_pcd = laspy.create()
            output_pcd.x = x
            output_pcd.y = y
            output_pcd.z = z

            if save_colours:
                output_pcd.red = red
                output_pcd.green = green
                output_pcd.blue = blue

            output_pcd.write(os.path.join(output_dir, f'{id}.laz'))