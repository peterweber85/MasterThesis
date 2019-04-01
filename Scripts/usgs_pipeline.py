
#%%
import gist

from PIL import Image
import imageio
from IPython.display import display, clear_output

import sys
sys.path.append("../Library/")
import image_download as imd
import image_manipulation as ima

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from dotenv import load_dotenv
import os
load_dotenv("../.env")
import csv

import numpy as np
import db_connection as dbcon
from datetime import datetime

#%%
# In parent directory, to get env variables
dotenv_path = ('../.env')
load_dotenv(dotenv_path)

#%% Parameters
# PROCESSING PARAMETERS
PROCESS_RAW_IMAGES = True
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
MFP_IMG_FOLDER = GDRIVE_FOLDER + 'MFP - Satellogic/images/'
CATEGORIES = ['city', 'agriculture', 'forest-woodland', 'semidesert', 'shrubland-grassland']

# Compute more parameters
params = {'size': SIZE, 'res': BASE_RESOLUTION, 'res_degr': DEGRADED_RESOLUTIONS}
subfolder_size = MFP_IMG_FOLDER + 'usgs_' + str(SIZE) + "/"
subfolder_base_res = subfolder_size  + "usgs_" + str(SIZE) + "_" + str(BASE_RESOLUTION) + "m/"

#%% Processing
ima.create_directory(subfolder_size)
ima.create_directory(subfolder_base_res)

for i in range(len(CATEGORIES)):
    # Process raw images and save
    if PROCESS_RAW_IMAGES:
        print("Processing raw images of category", CATEGORIES[i], "...")
        output_folder = subfolder_base_res + CATEGORIES[i] + "/"
        raw_images_fullnames = ima.list_path_of_images_by_category(RAW_IMAGE_FOLDER, CATEGORIES[i])
        imd.process_raw_images_and_save_usgs(raw_images_fullnames, params, output_folder)

    # Degrade images and save
    if DEGRADE_IMAGES:
        print("Degrading images of category", CATEGORIES[i], "...")
        images_fullnames = ima.list_path_of_images_by_category(subfolder_base_res, CATEGORIES[i])
        ima.degrade_images_and_save(images_fullnames, params, subfolder_size, CATEGORIES[i])

#%%
