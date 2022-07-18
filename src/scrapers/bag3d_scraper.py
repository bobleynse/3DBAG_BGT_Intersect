import requests
import urllib.request, json 
from cjio import cityjson

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from concurrent.futures import ThreadPoolExecutor, as_completed
import os.path
# from os import path

from src.utils.pointclouds import get_bbox_from_tile_codes

max_jobs = 10
skip_existing = True

#Make sure to use the latest version path (plucked from a download url like: https://3dbag.nl/en/download?tid=5893)
version_substring = "v210908_fd2cee53"
download_path = "https://data.3dbag.nl/cityjson/{}/3dbag_{}_{}.json"
local_filename = "{}_{}_bbox_{}_{}_{}_{}.json"
local_path = "C:/Users/boble/Documents/AI-year2/Thesis/data/"


def download_tile(url, file_name):
    try:
        #skip stuff we already downloaded
        if skip_existing and os.path.isfile(file_name):
            return "- " + file_name
            
        #download the download URL, with a delay and retry for when we get blocked
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=1.0)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        downloaded_obj = session.get(url, stream=True)
        
        #if downloaded data has enough content
        if len(downloaded_obj.content) > 15:
            open(file_name, 'wb').write(downloaded_obj.content)
            return "+ " + file_name
            
        return "Not CityJSON: " + url
        #return downloaded_obj.status_code
    except requests.exceptions.RequestException as exception:
       return exception


def parse_wfs_json(wfs_url):
    """_summary_

    Args:
        wfs_url (str): url to the containing bounding box

    Returns:
        list of str: containing all filepaths
    """
    filepaths = []

    #create the files its containing folder if it doenst exist
    if not os.path.exists(os.path.dirname(local_path)):
        try:
            os.makedirs(os.path.dirname(local_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise  

    with urllib.request.urlopen(wfs_url) as url:
        print(wfs_url)
        data = json.loads(url.read().decode())
        
        #Now do the downloads in multiple threads for speeeed
        threads = []
        with ThreadPoolExecutor(max_workers=max_jobs) as executor:
            print("Starting threads")
            print("Found features:" + str(data["totalFeatures"])) 
            for feature in data["features"]:   
                tile_id = feature["properties"]["tile_id"]
                cnt = feature["properties"]["cnt"]
                bbox_minx = feature["bbox"][0]
                bbox_miny = feature["bbox"][1]
                bbox_maxx = feature["bbox"][2]
                bbox_maxy = feature["bbox"][3]
                
                download_url = download_path.format(version_substring,version_substring,tile_id)
                print(feature["properties"]["tile_id"] + " -> " + download_url)

                filepath = local_path + local_filename.format(tile_id,cnt,bbox_minx,bbox_miny,bbox_maxx,bbox_maxy)
                threads.append(executor.submit(download_tile, download_path.format(version_substring,version_substring,tile_id), filepath))
                filepaths.append(filepath)
            for task in as_completed(threads):
                print(task.result())  
    return filepaths


def get_bag3d_as_json(tile_codes, padding=0.0):
    """
    Args:
        tile_codes (list of str): The tile codes, e.g. [2386_9702, 2446_9521].
        padding (float, optional): Add padding to bounding box. Defaults to 0.0.
    Returns:
        cjio
    """

    (bbox_min_x, bbox_max_y), (bbox_max_x, bbox_min_y) = get_bbox_from_tile_codes(tile_codes, padding=padding)

    wfs_url = f"https://data.3dbag.nl/api/BAG3D_v2/wfs?version=1.1.0&request=GetFeature&typename=BAG3d_v2:bag_tiles_3k&outputFormat=application/json&srsname=EPSG:28992&bbox={bbox_min_x},{bbox_min_y},{bbox_max_x},{bbox_max_y},EPSG:28992"

    # Download the files and save locally
    parsed_filedirs = parse_wfs_json(wfs_url)

    # Load the files and merge
    city_model = cityjson.load(parsed_filedirs[0])
    for filedir in parsed_filedirs[1:]:
        # TODO check if correct
        city_model.merge(cityjson.load(filedir))
    
    return city_model


def get_bag3d_building_by_id(bag_buildings, id, lod=2.2):
    """
    Args:
        bag_buildings (cityjson): cityjson filtered on buildings and building parts using get_cityobjects
        id (str): bag_id, number part only

    Returns:
        _type_: list containing the lod2.2 elements of all bag_id children
    """

    bag_id = f'NL.IMBAG.Pand.{id}'
    lod_geom_list = []

    for bag_child_id in bag_buildings[bag_id].children:
        bag_child = bag_buildings[bag_child_id]

        for lod_geom in bag_child.geometry:
            if lod_geom.lod == lod:
                lod_geom_list.append(lod_geom)
    return lod_geom_list


def get_surfaces(geometry, type='RoofSurface', lod=2.2):
    """Collect all surfaces of a geometry object

    Args:
        geometry_list (_type_): _description_
        type (str, optional): _description_. Defaults to 'RoofSurface'. WallSurface, GroundSurface

    Returns:
        list of vedo.Mesh'es
    """
    surfaces = geometry.surfaces
    boundaries = geometry.boundaries

    vedo_surfaces = []

    if type == 'RoofSurface':
        geometry_ids = [1]
    elif type == 'WallSurface':
        geometry_ids = [2, 3]
    elif type == 'GroundSurface':
        geometry_ids = [0]
    else:
        raise TypeError(f'Geometry type <{type}> not available')

    for geometry_id in geometry_ids:
        if surfaces[geometry_id]['surface_idx']:
            for element_id in surfaces[geometry_id]['surface_idx']:
                vedo_surfaces.append(boundaries[0][element_id[1]])
    return vedo_surfaces