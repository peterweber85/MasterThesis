
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
PROCESS_RAW_IMAGES = False
MOVE_PROCESSED_IMAGES = False
DEGRADE_IMAGES = False
SAVE_LABELS = True

# IMAGE PARAMETERS
SIZE = 1000 # in pixels
BASE_RESOLUTION = 0.3 # in meter
LABELS = [0, 1, 2]

# THESE ARE ONLY APPROXIMATE -->  integer(SIZE/DEGRADED_RESOLUTION)
DEGRADED_RESOLUTIONS = [0.6, 1, 2, 3, 5, 10, 15, 20, 30] # in meter

DEGRADED_SIZES = [SIZE/(res/BASE_RESOLUTION) for res in DEGRADED_RESOLUTIONS]
print("Degraded sizes are", DEGRADED_SIZES, "pixels. This is rounded to the nearest integer!")
# THESE ARE ONLY APPROXIMATE -->  integer(SIZE/DEGRADED_RESOLUTION)

# FOLDER PARAMETERS
GDRIVE_FOLDER = os.getenv('GDRIVE_FOLDER')
RAW_IMAGE_FOLDER = GDRIVE_FOLDER + 'MFP - Satellogic/images/raw_images_usgs_0.3m/'
PROCESSED_IMAGE_FOLDER = RAW_IMAGE_FOLDER + 'processed/'
MFP_IMG_FOLDER = GDRIVE_FOLDER + 'MFP - Satellogic/images/'
CATEGORIES = ['agriculture', 'shrubland-grassland', 'city', 'forest-woodland', 'semi-desert']

# Compute more parameters
params = {'size': SIZE, 'res': BASE_RESOLUTION, 'res_degr': DEGRADED_RESOLUTIONS}
subfolder_size = MFP_IMG_FOLDER + 'usgs_' + str(SIZE) + "/"
subfolder_base_res = subfolder_size + "usgs_" + str(SIZE) + "_" + str(BASE_RESOLUTION) + "m/"


#%% Processing

print("\nUSGS PIPELINE:\n")
t0 = time.time()
db = dbcon.connect("../credentials/mlab_db.txt", "mfp")
images_usgs_col = db["images_lib_usgs"]
labels_usgs_col = db["labels_lib_usgs"]
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
    #%%
    if SAVE_LABELS:
        print("\nSave labels of category", category.upper(), "to csv ...")
        t1 = time.time()
        ima.create_csv_with_labels_by_category_usgs(subfolder_base_res, category, LABELS)
        t2 = time.time()
        print("[{:.2f} s]".format(t2 - t1))

#%%
print("\nDONE! [{:.2f} s]\n".format(t2 - t0))

#%%


#%% DEPRECATED
def write_labels_from_csv_to_db_usgs(db_collection, folder_name, csv_filename):
    """

    :param db_collection: mongodb connection and define collection
    :param folder_name: str
    :param csv_filename: str
    :return:
    """
    with open(folder_name + csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print("loading labels from file " + csv_filename + " to db ...")
                line_count += 1
            else:
                query = {"filename": row[0]}
                label = {"$set": {"label": row[1]}}
                db_collection.update(query, label)
                line_count += 1
        print(str(line_count - 1) + ' labels added to db!')
    return