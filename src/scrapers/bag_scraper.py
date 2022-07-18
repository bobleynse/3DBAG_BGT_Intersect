import json
from owslib.wfs import WebFeatureService

from src.utils.pointclouds import get_bbox_from_tile_codes

WFS_URL = "https://data.3dbag.nl/api/BAG3D_v2/wfs"
WFS_LAYER = "BAG3D_v2:lod12"


def get_bag_by_tile_codes(tile_codes, padding=5):
    """
    Args:
        tile_codes (list of str): The tile codes, e.g. [2386_9702, 2446_9521].
        padding (float, optional): Add padding to bounding box. Defaults to 0.0.
    Returns:
        dict: containing an outline per id
    """

    parsed_content = {}

    (xmin, ymax), (xmax, ymin) = get_bbox_from_tile_codes(tile_codes, padding)

    wfs = WebFeatureService(url=WFS_URL, version='1.1.0')

    response = wfs.getfeature(
        typename=WFS_LAYER,
        bbox=[xmin, ymin, xmax, ymax],
        srsname='urn:x-ogc:def:crs:EPSG:28992',
        outputFormat='json'
    )

    j = json.loads(response.read().decode("utf-8"))

    # List of all buildings in this area
    for item in j['features']:
        id = item['properties']['identificatie'][14:]
        parsed_content[id] = {}
        parsed_content[id]['outline'] = item['geometry']['coordinates']
    return parsed_content