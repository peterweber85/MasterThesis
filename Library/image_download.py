
# GENERAL:
import math
import numpy as np
import scipy.optimize as so
from dotenv import load_dotenv
import os
import datetime

# MAP & IMG:
from motionless import CenterMap
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from urllib import request
import imageio

# Library
import db_connection as dbcon


# In file directory, to get google maps api key
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# In parent directory, to get google maps api key
dotenv_path = ('../.env')
load_dotenv(dotenv_path)


def zoom_in_meters_per_pixel(zoom, lat):
    """
    Convert zoom to meters per pixel at coordinate defined by lat
    :param zoom: int
        Google Maps zoom, 1 - 20
    :param lat: float
    :return:
    """
    meters_per_px = 156543.03392 * math.cos(lat * math.pi / 180) / 2 ** zoom
    return meters_per_px


def measure_distance(lat1, lon1, lat2, lon2):
    """
    returns distance in meters provided two lat/long coordinates

    :param lat1: float
    :param lon1: float
    :param lat2: float
    :param lon2: float
    :return:
    """
    radius = 6378.137  ## Radius of earth in KM
    dlat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dlon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(lat1 * math.pi / 180) * math.cos(
        lat2 * math.pi / 180) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c * 1000  ## meters
    return d


def get_dlat_dlong_from_distance(lat, long, distance):
    """
    returns difference in lat and in long for provided distance
    :param lat: float
    :param long: float
    :param distance: float
        distance in meter
    :return:
    """
    lat_fct = lambda x: abs(measure_distance(x, long, lat, long) - distance)
    long_fct = lambda y: abs(measure_distance(lat, y, lat, long) - distance)

    dlat = abs(so.minimize_scalar(lat_fct).x - lat)
    dlong = abs(so.minimize_scalar(long_fct).x - long)

    return dlat, dlong


def lat_long_array(lat, long, zoom, pixels, num_images, distance_factor=1, center=True, xy_to_ij = True):
    """
    generates array of lat/long coordinates of size num_images x num_images x 2

    :param lat: float
        lat coordinate of either central (center = True) image,
        or first image on the left (center = False)
    :param long: float
        long coordinate of either central (center = True) image,
        or first image on the left (center = False)
    :param zoom: int
        google maps api zoom
    :param pixels: int
        maximum of 640
    :param num_images: int
    :param distance_factor: int
        if 1 then generates array with images 'touching' each other
        if > 1 then generates array where images are separated by the factor
    :param center: bool
        see lat/long parameters
    :param xy_to_ij: bool
        if True then coordinates of array will be indexed as in matrix (left to right, up to down)
        if False then coordinates of array will be indexed as in cartesian coordinate system, first quadrant
            (left to right, down to up)
    :return:
    """
    ## Get distance between images
    meters_per_px = zoom_in_meters_per_pixel(zoom, lat)
    image_size_meters = meters_per_px * pixels

    ## Convert images distance into lat/long difference
    dlat, dlong = get_dlat_dlong_from_distance(lat, long, image_size_meters)

    ## Convert lat and long to vector
    if xy_to_ij:
        latvec = np.linspace(lat, lat - (num_images - 1) * dlat * distance_factor, num_images)
    else:
        latvec = np.linspace(lat, lat + (num_images - 1) * dlat * distance_factor, num_images)
    longvec = np.linspace(long, long + (num_images - 1) * dlong * distance_factor, num_images)
    ## convert provided lat/long to be central image
    if center:
        latvec = latvec - (np.mean(latvec) - latvec[0])
        longvec = longvec - (np.mean(longvec) - longvec[0])
    ## Convert both vectors to common array
    XY = np.array([[(latvec[i], longvec[j]) for j in range(longvec.shape[0])] for i in range(latvec.shape[0])])
    return XY


def generate_gmaps_links(lat, long, zoom, pixels, num_images, center=True, xy_to_ij=True):
    """
    generates list with google maps static api links

    paramter defintion as in function 'lat_long_array'
    """
    coord = lat_long_array(lat, long, zoom, pixels, num_images, center=center, xy_to_ij=xy_to_ij)

    urls = []
    for i in range(coord.shape[0]):
        for j in range(coord.shape[1]):
            cmap = CenterMap(lat=coord[i, j, 0],
                             lon=coord[i, j, 1],
                             maptype='satellite',
                             size_x=pixels,
                             size_y=pixels,
                             zoom=zoom,
                             key=os.getenv("GMAPS_API_KEY")
                             )
            urls.append(cmap.generate_url())
    return urls

def download_image(lat, lon, zoom, pixels, gmaps_key, folder='', plot_image=False):
    """
    Downloads and returns (optionally plots) an image from Google Maps static API
    :param lat: float
        central lat coordinate
    :param lon: float
        central lon coordinate
    :param zoom: int
        google maps api zoom
    :param pixels: int
        maximum of 640
    :param gmaps_key: string
        Google Maps API key
    :param folder: string
        where to save the image
    :param plot_image: bool
        whether to plot the image
    :return:
    """

    img_metadata = {}
    img_metadata["lat"] = lat
    img_metadata["lon"] = lon
    img_metadata["zoom"] = zoom
    img_metadata["pixels"] = pixels

    meters_per_px = zoom_in_meters_per_pixel(zoom, lat)
    image_size = meters_per_px * pixels
    img_metadata["meters_per_px"] = meters_per_px
    img_metadata["img_size"] = image_size

    # download image
    url = CenterMap(lat=lat, lon=lon, maptype='satellite', size_x=pixels, size_y=pixels, zoom=zoom, key=gmaps_key).generate_url()
    img_metadata["url"] = url
    image = Image.open(BytesIO(request.urlopen(url).read()))

    img_metadata["filename"] = str(lat) + '_' + str(lon) + '_' + str(zoom) + '_' + str(pixels) + '.png'
    image.save(folder + img_metadata["filename"], "PNG")
    img_metadata["saved_dt"] = datetime.datetime.today()

    if plot_image:
        image.show()
        plt.show()

    return img_metadata


def download_and_save_image(name, lat, lon, zoom, pixels, gmaps_key, folder='', save_image = True):
    """
    Downloads and returns an image from Google Maps static API
    :param lat: float
        central lat coordinate
    :param lon: float
        central lon coordinate
    :param zoom: int
        google maps api zoom
    :param pixels: int
        maximum of 640
    :param gmaps_key: string
        Google Maps API key
    :param folder: string
        where to save the image
    :param save_image: bool
        whether to save the image
    :return:
    """
    url = CenterMap(lat=lat, lon=lon, maptype='satellite', size_x=pixels, size_y=pixels, zoom=zoom,
                    key=gmaps_key).generate_url()
    image = Image.open(BytesIO(request.urlopen(url).read()))
    fname_suffix = name + '_' + str(lat) + '_' + str(lon) + '_' + str(zoom) + '_' + str(pixels) + '.png'
    if save_image:
        image.save(folder + fname_suffix, "PNG")
    return image

def generate_metadata(name, lat, lon, zoom, pixels, gmaps_key):
    """
    Generates metadata of provided parameters for image
    :param name: string
        name identifier of the image group, e.g. name of city
    :param label: int
        one of 0,1,2,3,4 where 0 maximum natural and 4 maximum man-made
    :param lat: float
        central lat coordinate
    :param lon: float
        central lon coordinate
    :param zoom: int
        google maps api zoom
    :param pixels: int
        maximum of 640
    :param gmaps_key: string
        Google Maps API key
    :return:
    """
    img_metadata = {}
    img_metadata["name"] = name
    img_metadata["lat"] = lat
    img_metadata["lon"] = lon
    img_metadata["zoom"] = zoom
    img_metadata["pixels"] = pixels
    meters_per_px = zoom_in_meters_per_pixel(zoom, lat)
    image_size = meters_per_px * pixels
    img_metadata["meters_per_px"] = meters_per_px
    img_metadata["img_size"] = image_size
    url = CenterMap(lat=lat, lon=lon, maptype='satellite', size_x=pixels, size_y=pixels, zoom=zoom,
                    key=gmaps_key).generate_url()
    img_metadata["url"] = url
    img_metadata["filename"] = name + '_' + str(lat) + '_' + str(lon) + '_' + str(zoom) + '_' + str(pixels) + '.png'
    img_metadata["saved_dt"] = datetime.datetime.today()
    return img_metadata


def download_images_random_gaussian(locations, zoom, pixels, samples_per_location, precision,
                                    api_key, img_folder, save_image=True):
    """

    :param locations: dict
     with keys: 'name', 'lat', 'lon'
    :param zoom: int
    :param pixels: int
    :param samples_per_location: int
    :param precision: int
        roundinf precision when choosing random location
    :param api_key: str
    :param img_folder: str
    :param save_image: bool
    :return:
    """
    images = []
    mdata = []

    db = dbcon.connect("../credentials/mlab_db.txt", "mfp")
    images_lib_col = db["images_lib"]

    for location in locations:
        print("Saving images of " + location["name"] + "...")
        for i in range(samples_per_location):
            lat = round(location["lat"] + np.random.normal(0, 0.1), precision)
            lon = round(location["lon"] + np.random.normal(0, 0.1), precision)
            image = download_and_save_image(location['name'], lat, lon, zoom, pixels, api_key,
                                                folder=img_folder, save_image=save_image)
            images.append(image)
            if save_image:
                metadata = generate_metadata(location['name'], lat, lon, zoom, pixels, api_key)
                mdata.append(metadata)
                images_lib_col.replace_one({"filename": metadata["filename"]}, metadata, upsert=True)

    return images, mdata


def download_images_defined_location(locations, zoom, pixels, center, xy_to_ij, num_images,
                                     api_key, img_folder, distance_factor=1, save_image=True):
    """
    Returns num_images ** 2 at locations defined by locations and array generated by lat_long_array()
    
    :param locations: dict
     with keys: 'name', 'lat', 'lon'
    :param zoom: int
    :param pixels: int
    :param center: bool
    :param xy_to_ij: bool
        if True then coordinates of array will be indexed as in matrix (left to right, up to down)
        if False then coordinates of array will be indexed as in cartesian coordinate system, first quadrant
            (left to right, down to up)
    :param num_images: int
        int**2 number of images will be returned
    :param api_key: str
    :param img_folder: str
    :param distance_factor: int
        if 1 then generates array with images 'touching' each other
        if > 1 then generates array where images are separated by the factor
    :param center: bool
        see lat/long parameters
    :param save_image: bool
    :return:
    """
    images = []
    mdata = []

    db = dbcon.connect("../credentials/mlab_db.txt", "mfp")
    images_lib_col = db["images_lib"]

    for location in locations:
        print("Saving images of " + location["name"] + "...")

        coord = lat_long_array(location['lat'],
                               location['lon'],
                               zoom, pixels, num_images,
                               distance_factor=distance_factor,
                               center=center,
                               xy_to_ij=xy_to_ij)
        for i in range(coord.shape[0]):
            for j in range(coord.shape[1]):
                lat = coord[i, j, 0]
                lon = coord[i, j, 1]
                image = download_and_save_image(location['name'], lat, lon, zoom, pixels, api_key,
                                                folder=img_folder, save_image=save_image)
                images.append(image)
                if save_image:
                    metadata = generate_metadata(location['name'], lat, lon, zoom, pixels, api_key)
                    mdata.append(metadata)
                    images_lib_col.replace_one({"filename": metadata["filename"]}, metadata, upsert=True)

    return images, mdata



def generate_random_location_in_rectangle(lat_tpl, lon_tpl):
    """
    Returns (lat, lon) tuple of random location determined by the borders the input tuples
    :param lat_tpl: tuple
    :param lon_tpl: tuple
    :return:
    """
    lat = np.random.uniform(lat_tpl[0], lat_tpl[1])
    lon = np.random.uniform(lon_tpl[0], lon_tpl[1])
    return lat, lon


def download_save_images_in_random_rectangle(db_collection,
                                              name,
                                              lat_tpl,
                                              lon_tpl,
                                              num_locations,
                                              zoom_levels,
                                              pixels,
                                              api_key,
                                              img_folder):
    """
    Generates num_locations random locations within rectangle defined by lat_tpl, lon_tpl
    Downloads images at these locations and stores metadata in db
    It does this for all the zoom levels

    :param db_collection:
    :param name: str
        name used to identify image group when later labelling
    :param lat_tpl: tuple of floats
    :param lon_tpl: tuple of floats
    :param num_locations: int
        number of different locations to download images
    :param zoom_levels:
    :param pixels:
    :param api_key:
    :param img_folder:
    :return:
    """
    for _ in range(num_locations):
        lat, lon = generate_random_location_in_rectangle(lat_tpl, lon_tpl)
        print("Coordinates: ", (lat, lon))
        for zoom in zoom_levels:
            print("Zoom: ", zoom)
            download_and_save_image(name, lat, lon, zoom, pixels, api_key,
                                    folder=img_folder, save_image=True)
            metadata = generate_metadata(name, lat, lon, zoom, pixels, api_key)
            db_collection.replace_one({"filename": metadata["filename"]}, metadata, upsert=True)
            print("Image and Metadata with filename '"+metadata["filename"]+"' saved!\n")