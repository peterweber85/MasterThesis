
#%% IMPORTS

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from datetime import datetime
import time

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
SIZE = 512 # in pixels
BASE_RESOLUTION = 1 # in meter

# THESE ARE ONLY APPROXIMATE -->  integer(SIZE/DEGRADED_RESOLUTION)
DEGRADED_RESOLUTIONS = [2, 5, 10, 20, 50] # in meter
# THESE ARE ONLY APPROXIMATE -->  integer(SIZE/DEGRADED_RESOLUTION)

# FOLDER PARAMETERS
GDRIVE_FOLDER = os.getenv('GDRIVE_FOLDER')
RAW_IMAGE_FOLDER = GDRIVE_FOLDER + 'MFP - Satellogic/images/raw_images_usgs/'
PROCESSED_IMAGE_FOLDER = RAW_IMAGE_FOLDER + 'processed/'
MFP_IMG_FOLDER = GDRIVE_FOLDER + 'MFP - Satellogic/images/'
CATEGORIES = ['city', 'agriculture', 'forest-woodland', 'semi-desert', 'shrubland-grassland']

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

for category in CATEGORIES:
    # Process raw images and save
    if PROCESS_RAW_IMAGES:
        print("\nProcessing raw images of category", category.upper(), "...")
        t1 = time.time()
        output_folder = subfolder_base_res + category + "/"
        raw_images_fullnames = ima.list_path_of_images_by_category(RAW_IMAGE_FOLDER, category)
        imd.process_raw_images_and_save_usgs(raw_images_fullnames, params, category, output_folder, images_usgs_col)
        if MOVE_PROCESSED_IMAGES:
            ima.move_folder_content(RAW_IMAGE_FOLDER + category, PROCESSED_IMAGE_FOLDER + category)
        t2 = time.time()
        print("[{:.2f} s]".format(t2 - t1))

    # Degrade images and save
    if DEGRADE_IMAGES:
        print("\nDegrading images of category", category.upper(), "...")
        t1 = time.time()
        images_fullnames = ima.list_path_of_images_by_category(subfolder_base_res, category)
        ima.degrade_images_and_save(images_fullnames, params, subfolder_size, category, images_usgs_col)
        t2 = time.time()
        print("[{:.2f} s]".format(t2 - t1))

print("\nDONE! [{:.2f} s]\n".format(t2 - t0))
