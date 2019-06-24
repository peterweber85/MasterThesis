
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
from PIL import Image, ImageOps
from urllib import request
import imageio
import gist

# Library
import image_manipulation as ima
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


def download_image(lat, lon, zoom, pixels, gmaps_key, crop_px=20, name='', dataset='', folder='', plot_image=False, save_image=True):
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
    :param crop_px: int
        number of pixels to crop for each border
    :param name: str
        name identifier of the image group, e.g. name of city
    :param dataset: str
        dataset identifier
    :param folder: string
        where to save the image
    :param plot_image: bool
        whether to plot the image
    :param save_image: bool
        whether to save the image
    :return:
    """

    metadata = {}
    metadata["lat"] = lat
    metadata["lon"] = lon
    metadata["zoom"] = zoom
    metadata["pixels"] = pixels
    metadata["name"] = name
    metadata["dataset"] = dataset

    meters_per_px = zoom_in_meters_per_pixel(zoom, lat)
    image_size = meters_per_px * pixels
    metadata["meters_per_px"] = meters_per_px
    metadata["img_size"] = image_size

    # download image
    url = CenterMap(lat=lat, lon=lon, maptype='satellite', size_x=pixels+2*crop_px, size_y=pixels+2*crop_px, zoom=zoom, key=gmaps_key).generate_url()
    metadata["url"] = url
    image = Image.open(BytesIO(request.urlopen(url).read()))
    image = ImageOps.crop(image, (crop_px, crop_px, crop_px, crop_px)) # borders: left, up, right, bottom

    metadata["filename"] = name + '_' + str(lat) + '_' + str(lon) + '_' + str(zoom) + '_' + str(pixels) + '.png'
    if save_image:
        image.save(folder + metadata["filename"], "PNG")
    metadata["saved_dt"] = datetime.datetime.today()

    if plot_image:
        image.show()
        plt.show()

    return image, metadata


def download_and_save_image(name, lat, lon, zoom, pixels, gmaps_key, folder='', save_image=True):
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


def generate_metadata_gmaps(name, lat, lon, zoom, pixels, gmaps_key):
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


def download_images_random_gaussian(location, samples_per_location, sd, coord_precision, zooms, pixels,
                                    api_key, dataset, img_folder, db_col, plot_image=False, save_image=True):
    """
    :param location: dict
     with keys: 'name', 'lat', 'lon'
    :param samples_per_location: int
    :param sd: float
    :param coord_precision: int
        rounding precision when choosing random location
    :param zooms: list of int
    :param pixels: int
    :param api_key: str
    :param dataset: str
    :param img_folder: str
    :param db_col: Collection
    :param plot_image: bool
    :param save_image: bool
    :return:
    """

    images = []
    mdata = []

    for i in range(samples_per_location):
        lat = round(location["lat"] + np.random.normal(0, sd), coord_precision)
        lon = round(location["lon"] + np.random.normal(0, sd), coord_precision)
        for zoom in zooms:
            image, metadata = download_image(
                lat, lon, zoom, pixels, api_key,
                crop_px=20, name=location["name"], dataset=dataset, folder=img_folder,
                plot_image=plot_image, save_image=save_image
            )
            images.append(image)
            mdata.append(metadata)
            db_col.replace_one({"filename": metadata["filename"]}, metadata, upsert=True)

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
                    metadata = generate_metadata_gmaps(location['name'], lat, lon, zoom, pixels, api_key)
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
            metadata = generate_metadata_gmaps(name, lat, lon, zoom, pixels, api_key)
            db_collection.replace_one({"filename": metadata["filename"]}, metadata, upsert=True)
            print("Image and Metadata with filename '"+metadata["filename"]+"' saved!\n")


def get_image_grid(imarray, size, include_last = False):
    """
    From large image obtain a grid to crop several smaller images.
    It also centers the grid.

    :param imarray: np.array
        of large image
    :param size: int
        distance of grid elements
    :return: dict
        keys are grid indices
        values are grid coordinates
    """

    dim = imarray.shape[:2]

    # Number of images along every axis
    num_x = int((dim[1] - size) / size) + 1
    num_y = int((dim[0] - size) / size) + 1

    # Center images
    x_move_by = int((dim[1] - num_x * size) / 2)
    y_move_by = int((dim[0] - num_y * size) / 2)

    # include_last is only needed for presentation purposes when plotting the grid
    if include_last: num_x, num_y = num_x + 1, num_y + 1

    # create dictionary of grid points
    grid = dict()
    for ix in range(num_x):
        for iy in range(num_y):
            grid[(iy, ix)] = (size * iy + y_move_by, size * ix + x_move_by)
    return grid


def get_cropped_images(imarray, grid):
    """
    Returns several cropped images from one large image provided a grid
    :param imarray: np.array
        of large image
    :param grid: dict
        Output of get_image_grid with
            keys are grid indices
            values are grid coordinates
    :return: dict
        keys are coordinate of image
        values are image
    """
    x_coord = [x[0] for x in list(grid.values())]
    size = x_coord[1] - x_coord[0]

    output = dict()
    for coord in grid.values():
        img = imarray[coord[0]:coord[0] + size, coord[1]:coord[1] + size]
        output[coord] = img

    return output


def generate_metadata_usgs(category, orig_img, filename, coordinate, size, res, gist_vector, dataset):
    """
    Generates metadata of provided parameters for image
    :param category: string
        name identifier of the image group/category
    :param orig_img: string
        filename of the original, complete image
    :param filename: string
        filename of the cropped image
    :param coordinate: list
        x,y coordinates of the cropped image
    :param size: int
        images size, in pixels
    :param res: int
        image resolution, in meter
    :param gist_vector:
        gist vector of the image
    :param dataset: string
        dataset name identifier
    :return:
    """
    img_metadata = {}
    img_metadata["dataset"] = dataset
    img_metadata["category"] = category
    img_metadata["original_image"] = orig_img
    img_metadata["filename"] = filename
    img_metadata["coordinate_x"] = coordinate[0]
    img_metadata["coordinate_y"] = coordinate[1]
    img_metadata["size"] = size
    img_metadata["res"] = res
    img_metadata["gist_"+str(res)] = gist_vector
    img_metadata["saved_dt"] = datetime.datetime.today()
    return img_metadata


def save_cropped_images(imcropped, params, input_fname, category, output_folder, db_collection = None):
    """
    :param imcropped: dict
        Output of get_cropped_images
            keys are coordinate of image
            values are image
    :param params: dict
        parameter with image properties
        params['size'] = size
        params['res'] = resolution
    :param input_fname: str
        filename of large image
    :param category: str
        name identifier of the image group/category
    :param output_folder: str
    :param db_collection: mongodb collection object
    :return:
    """
    size = params['size']
    baseres = params['res']

    for coordinate in imcropped.keys():
        imarray = imcropped[coordinate]
        filename_pure = input_fname.split(".")[0]
        coordinate_string = "_x" + str(coordinate[0]) + "_y" + str(coordinate[1])
        size_string = "_size" + str(size)
        baseres_string = "_baseres" + str(baseres) + "m"
        filename = filename_pure + coordinate_string + size_string + baseres_string + ".png"
        output_path = os.path.join(output_folder, filename)
        Image.fromarray(imarray).save(output_path)

        if not db_collection is None:
            gist_vector = gist.extract(imarray, nblocks=1, orientations_per_scale=(8, 8, 4)).tolist()
            metadata = generate_metadata_usgs(
                category, filename_pure, filename, coordinate, size, baseres,
                gist_vector, dataset='usgs' + '_res' + str(baseres) + "m" + '_size' + str(size))
            db_collection.replace_one({"filename": metadata["filename"]}, metadata, upsert=True)

def process_raw_images_and_save_usgs(paths, params, category, output_folder, db_collection = None):
    """
    Processing function that combines
        - loading image
        - obtaining grid for cropping
        - cropping small images from large image
        - saving cropped images
    :param paths: list of str
    :param params: dict
        parameter with image properties
        params['size'] = size
        params['res'] = resolution
    :param category: str
        name identifier of the image group/category
    :param output_folder: str
    :param db_collection: mongodb collection object
    :return:
    """
    if isinstance(paths, str):
        paths = [paths]

    ima.create_directory(output_folder)

    for path in paths:
        filename = path.split("/")[-1]
        print("\n-------------------------------------------- Processing image", filename, "!!! --------------------------------------------")
        imarray = ima.load_image_as_rgb_array(path)
        grid = get_image_grid(imarray, params['size'])
        imcropped = get_cropped_images(imarray, grid)
        save_cropped_images(imcropped, params, filename, category, output_folder)
