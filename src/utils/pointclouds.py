import numpy as np


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