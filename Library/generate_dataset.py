
# GENERAL:
#import math
#import numpy as np
#import scipy.optimize as so
#from dotenv import load_dotenv
import os
#import datetime

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
    "per_city": 10,
    "per_capital": 10,
    "per_country": 10,
}

ZOOM_LEVELS = [13,14,15,16]

PIXELS = 600

if __name__ == '__main__':

    # connect to db and collection (to pass it as a parameter of each function)
    db = dbcon.connect("../credentials/mlab_db.txt", "mfp")
    images_lib_col = db["images_lib"]

    # download images cities
    for city in coord.cities:
        pass

    # downlaod images random rectangle

    # download images capitals?
    # download images centroids
    # download random images around world?
    #
