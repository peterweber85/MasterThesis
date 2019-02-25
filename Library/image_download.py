
import math
import numpy as np
import scipy.optimize as so
from motionless import CenterMap
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from urllib import request

from dotenv import load_dotenv
import os

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
                             format='JPEG',
                             key=os.getenv("GMAPS_API_KEY")
                             )
            urls.append(cmap.generate_url())
    return urls


def download_images(links, plot_images=False):
    """
    Downloads and returns (optionally plots) images from google maps static api links
    :param links: list of strings
        output of generate_gmaps_links
    :param plot_images: bool
        whether to plot the images, be careful when downloading many images, it will plot all
    :return:
    """

    images = []
    for url in links:
        buffer = BytesIO(request.urlopen(url).read())
        image = Image.open(buffer)
        images.append(image)
        if plot_images:
            image.show()
    if plot_images:
        plt.show()

    return images