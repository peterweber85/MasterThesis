
#%% IMPORTS

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from datetime import datetime
import time
import csv
import pymongo

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

# IMAGE PARAMETERS
SIZE = 512 # in pixels
BASE_RESOLUTION = 1 # in meter

# FOLDER PARAMETERS
GDRIVE_FOLDER = os.getenv('GDRIVE_FOLDER')
RAW_IMAGE_FOLDER = GDRIVE_FOLDER + 'MFP - Satellogic/images/raw_images_usgs_1m/'
PROCESSED_IMAGE_FOLDER = RAW_IMAGE_FOLDER + 'processed/'
MFP_IMG_FOLDER = GDRIVE_FOLDER + 'MFP - Satellogic/images/'
CATEGORIES = ['agriculture', 'semi-desert', 'shrubland-grassland', 'forest-woodland']

# Compute more parameters
params = {'size': SIZE, 'res': BASE_RESOLUTION}
subfolder = os.path.join(MFP_IMG_FOLDER, 'usgs_' + str(SIZE) + '_res' + str(BASE_RESOLUTION) + 'm')


#%% Processing

print("\nUSGS PIPELINE:\n")
t0 = time.time()
ima.create_directory(subfolder)

for category in CATEGORIES:
    # Process raw images and save
    if PROCESS_RAW_IMAGES:
        print("\nProcessing raw images of category", category.upper(), "...")
        t1 = time.time()
        output_folder = os.path.join(subfolder, category)
        raw_images_fullnames = ima.list_path_of_images_by_category_and_label(RAW_IMAGE_FOLDER, category)
        imd.process_raw_images_and_save_usgs(raw_images_fullnames, params, category, output_folder)
        if MOVE_PROCESSED_IMAGES:
            ima.move_folder_content(RAW_IMAGE_FOLDER + category, PROCESSED_IMAGE_FOLDER + category)
        t2 = time.time()
        print("[{:.2f} s]".format(t2 - t1))

#%%
print("\nDONE! [{:.2f} s]\n".format(t2 - t0))

#%%