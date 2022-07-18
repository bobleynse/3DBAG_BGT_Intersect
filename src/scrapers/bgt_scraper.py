from re import L
import requests
from src.utils.pointclouds import get_bbox_from_tile_codes


def scrape_amsterdam_bgt(layer_name, bbox=None):
    """
    Scrape BGT layer information from the WFS.

    Parameters
    ----------
    layer_name : str
        Information about the different layers can be found at:
        https://www.amsterdam.nl/stelselpedia/bgt-index/producten-bgt/prodspec-bgt-dgn-imgeo/

    Returns
    -------
    The WFS response in JSON format or a dict.
    """
    params = 'REQUEST=GetFeature&' \
             'SERVICE=wfs&' \
             'VERSION=2.0.0&' \
             'TYPENAME=' \
             + layer_name + '&'

    WFS_URL = 'https://map.data.amsterdam.nl/maps/bgtobjecten?'

    if bbox is not None:
        bbox_string = str(bbox[0][0]) + ',' + str(bbox[0][1]) + ',' \
                      + str(bbox[1][0]) + ',' + str(bbox[1][1])
        params = params + 'BBOX=' + bbox_string + '&'

    params = params + 'OUTPUTFORMAT=geojson'

    response = requests.get(WFS_URL + params)
    try:
        return response.json()
    except ValueError:
        return None


def parse_polygons(json_response, include_bbox=True):
    """
    Parse the JSON content and transform it into a table structure.

    Parameters
    ----------
    json_response : dict
        JSON response from a WFS request.
    include_bbox : bool (default: True)
        Whether to include a bounding box for each poly.
    """
    parsed_content = {}
    # name = '_'.join(json_response['name'].split('_')[2:])
    for item in json_response['features']:
        # Add an item per bag id
        id = item['properties']['identificatieBAGPND']
        parsed_content[id] = {}

        # Add floorplan
        floorplan = item['geometry']['coordinates']
        parsed_content[id]['floorplan'] = floorplan

        if include_bbox:
            parsed_content[id]['bbox'] = None

    return parsed_content


# def get_bgt_by_id(id):
#     # TODO fix this temp fix
#     bgt_floorplans = get_bgt_by_tile_codes(['2445_9723', '2450_9726'])
#     return bgt_floorplans[id]['floorplan']


def get_bgt_by_tile_codes(tile_codes, padding=0.0):
    """
    Args:
        tile_codes (list of str): The tile codes, e.g. [2386_9702, 2446_9521].
        padding (float, optional): Add padding to bounding box. Defaults to 0.0.
    Returns:
        dict: containing an floorplan per id
    """

    parsed_content = {}

    # Specify the bounding box we want to work with
    tile_bbox = get_bbox_from_tile_codes(tile_codes, padding=padding)
    parsed_content['tile_bbox'] = tile_bbox

    # Scrape data from the Amsterdam WFS, this will return a json response.
    json_response = scrape_amsterdam_bgt('BGT_PND_pand', bbox=tile_bbox)

    # Parse the downloaded json response.
    parsed_content = parse_polygons(json_response)

    return parsed_content