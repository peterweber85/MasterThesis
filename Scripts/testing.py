
#%% IMPORTS

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from datetime import datetime
import time
from PIL import Image

import sys
sys.path.append("../Library/")
import image_download as imd
import image_manipulation as ima
import db_connection as dbcon

from dotenv import load_dotenv
load_dotenv("../.env")
dotenv_path = ('../.env')
load_dotenv(dotenv_path)


#%% PARAMETERS

# PROCESSING PARAMETERS
PROCESS_RAW_IMAGES = True
MOVE_PROCESSED_IMAGES = True
DEGRADE_IMAGES = True

# IMAGE PARAMETERS
SIZE = 1000 # in pixels
BASE_RESOLUTION = 0.3 # in meter

# THESE ARE ONLY APPROXIMATE -->  integer(SIZE/DEGRADED_RESOLUTION)
DEGRADED_RESOLUTIONS = [0.6, 1, 2, 5, 10, 15, 20, 30, 50] # in meter

DEGRADED_SIZES = [SIZE/(res/BASE_RESOLUTION) for res in DEGRADED_RESOLUTIONS]
print("Degraded sizes are", DEGRADED_SIZES, "pixels. This is rounded to the nearest integer!")
# THESE ARE ONLY APPROXIMATE -->  integer(SIZE/DEGRADED_RESOLUTION)

# FOLDER PARAMETERS
GDRIVE_FOLDER = os.getenv('GDRIVE_FOLDER')
RAW_IMAGE_FOLDER = GDRIVE_FOLDER + 'MFP - Satellogic/images/raw_images_usgs_0.3m/'
PROCESSED_IMAGE_FOLDER = RAW_IMAGE_FOLDER + 'processed/'
MFP_IMG_FOLDER = GDRIVE_FOLDER + 'MFP - Satellogic/images/'
CATEGORIES = ['city', 'agriculture', 'forest-woodland', 'semidesert', 'shrubland-grassland']

# Compute more parameters
params = {'size': SIZE, 'res': BASE_RESOLUTION, 'res_degr': DEGRADED_RESOLUTIONS}
subfolder_size = MFP_IMG_FOLDER + 'usgs_' + str(SIZE) + "/"
subfolder_base_res = subfolder_size + "usgs_" + str(SIZE) + "_" + str(BASE_RESOLUTION) + "m/"


#%% Processing

print("\nUSGS PIPELINE:\n")
t0 = time.time()
db = dbcon.connect("../credentials/mlab_db.txt", "mfp")
images_usgs_col = db["images_lib_usgs"]
ima.create_directory(subfolder_size)
ima.create_directory(subfolder_base_res)


#%%

fullnames = ['/Users/peterweber/Google Drive/MFP - Satellogic/images/raw_images_usgs_0.3m/forest-woodland/17sku925565.tif',
             '/Users/peterweber/Google Drive/MFP - Satellogic/images/raw_images_usgs_0.3m/forest-woodland/19tdl080220.tif']

images= {}
for name in fullnames:
    fname = name.split("/")[-1]
    images[fname] = ima.load_image_as_rgb_array(name)