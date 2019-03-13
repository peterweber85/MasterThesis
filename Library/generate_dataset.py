
# GENERAL:
#import math
import numpy as np
#import scipy.optimize as so
#from dotenv import load_dotenv
import os
#import datetime
import time

# MAP & IMG:
#from motionless import CenterMap
#import matplotlib.pyplot as plt
#from io import BytesIO
#from PIL import Image
#from urllib import request
#import imageio

# Library
import db_connection as dbcon
import coordinates as coord
import image_download as imd



# In file directory, to get google maps api key
# dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
# load_dotenv(dotenv_path)

# In parent directory, to get google maps api key
# dotenv_path = ('../.env')
# load_dotenv(dotenv_path)


GMAPS_API_KEY = os.getenv("GMAPS_API_KEY")
IMG_FOLDER = os.environ["MFP_IMG_FOLDER"]

# number of images to be download for each dataset
NUM_IMAGES = {
    "per_city": 0,
    "per_capital": 0,
    "per_centroid": 0,
}

SD = 0.1 # standard deviation of gaussian samples
COORDINATES_PRECISION = 4
ZOOM_LEVELS = [13]#,14,15,16]
PIXELS = 600

if __name__ == '__main__':

    print("GENERATE DATASET:")
    NUM_IMAGES_DOWNLOADED = 0
    t0 = time.time()

    # connect to db and collection (to pass it as a parameter of each function)
    db = dbcon.connect("../credentials/mlab_db.txt", "mfp")
    images_lib_col = db["images_lib"]

    # download images CITIES
    print("\nSaving images of CITIES:")
    t1 = time.time()
    for city in coord.CITIES:
        print("  " + city["name"] + "...")
        images, mdata = imd.download_images_random_gaussian(
            city, NUM_IMAGES["per_city"], SD, COORDINATES_PRECISION, ZOOM_LEVELS, PIXELS,
            GMAPS_API_KEY, IMG_FOLDER, images_lib_col, plot_image=False, save_image=True)
        NUM_IMAGES_DOWNLOADED += len(mdata)
    t2 = time.time()
    print("time: {:.2f} s".format(t2-t1))

    # download images CAPITALS
    print("\nSaving images of CAPITALS:")
    t1 = time.time()
    for capital in coord.CAPITALS:
        capital["name"] = capital["CapitalName"]
        capital["lat"] = float(capital["CapitalLatitude"])
        capital["lon"] = float(capital["CapitalLongitude"])
        print("  " + capital["name"] + "...")
        images, mdata = imd.download_images_random_gaussian(
            capital, NUM_IMAGES["per_capital"], SD, COORDINATES_PRECISION, ZOOM_LEVELS, PIXELS,
            GMAPS_API_KEY, IMG_FOLDER, images_lib_col, plot_image=False, save_image=True)
        NUM_IMAGES_DOWNLOADED += len(mdata)
    t2 = time.time()
    print("time: {:.2f} s".format(t2 - t1))

    # download images CENTROIDS
    print("\nSaving images of CENTROIDS:")
    t1 = time.time()
    for centroid in coord.CENTROIDS:
        centroid["name"] = centroid["CountryName"]
        centroid["lat"] = centroid["Latitude"]
        centroid["lon"] = centroid["Longitude"]
        print("  " + centroid["name"] + "...")
        images, mdata = imd.download_images_random_gaussian(
            centroid, NUM_IMAGES["per_centroid"], 10, COORDINATES_PRECISION, ZOOM_LEVELS, PIXELS,
            GMAPS_API_KEY, IMG_FOLDER, images_lib_col, plot_image=False, save_image=True)
        NUM_IMAGES_DOWNLOADED += len(mdata)
    t2 = time.time()
    print("time: {:.2f} s".format(t2 - t1))

    # download images random rectangle

    # download images capitals?
    # download images centroids
    # download random images around world?
    #

    t2 = time.time()
    print("\nDONE!")
    print(str(NUM_IMAGES_DOWNLOADED) + " images downloaded")
    print("TOTAL TIME: {:.2f} s".format(t2 - t0))
    print()
