
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
CREATE_LABEL_FOLDERS = True
LABEL_FROM_FOLDER = False
UPLOAD_LABEL_TO_DB = False

# FOLDER PARAMETERS
GDRIVE_FOLDER = os.getenv('GDRIVE_FOLDER')
MFP_IMG_FOLDER = GDRIVE_FOLDER + 'MFP - Satellogic/images/'
BASE_FOLDER = 'usgs_512/usgs_512_1m/'
LABELS_FOLDER = MFP_IMG_FOLDER + 'labels/'
CATEGORIES = ['city', 'agriculture', 'forest-woodland', 'semi-desert', 'shrubland-grassland']
LABEL_FOLDERS = ['label_0','label_1','label_2']

# DF PARAMETERS
label_name = 'label'
df_fields = ['filename', 'category', label_name]

#%% Processing

print("\nUSGS LABELS FROM FOLDERS:\n")
t0 = time.time()
db = dbcon.connect("../credentials/mlab_db.txt", "mfp")
images_usgs_col = db["images_lib_usgs"]
labels_df = pd.DataFrame(columns=df_fields)
labels_list = []

if CREATE_LABEL_FOLDERS:
    for category in CATEGORIES:
        for label_folder in LABEL_FOLDERS:
            ima.create_directory(MFP_IMG_FOLDER + BASE_FOLDER + '/' + category + '/' + label_folder)

if LABEL_FROM_FOLDER:
    for category in CATEGORIES:
        # Get labels from category folder
        category_folder = BASE_FOLDER + '/' + category + '/'
        print("\nLabelling images in folder", category_folder, "...")
        t1 = time.time()
        for label_folder in LABEL_FOLDERS:
            label = label_folder.split("_")[-1]
            #filenames = list_image_files_from_folder(MFP_IMG_FOLDER+category_folder+label_folder)
            filenames = ["test1.png","test2.png"]
            for filename in filenames:
                labels_list.append({
                    "filename": filename,
                    "category": category,
                    label_name: label
                })

        #df_labels_category = pd.DataFrame.from_dict(...)
        #df_labels = df_labels.append(df_labels_category, ignore_index=True, sort=False)

        #output_folder = subfolder_base_res + category + "/"
        #raw_images_fullnames = ima.list_path_of_images_by_category(RAW_IMAGE_FOLDER, category)
        #imd.process_raw_images_and_save_usgs(raw_images_fullnames, params, category, output_folder, images_usgs_col)
        #if MOVE_PROCESSED_IMAGES:
        #    ima.move_folder_content(RAW_IMAGE_FOLDER + category, PROCESSED_IMAGE_FOLDER + category)
        t2 = time.time()
        print("[{:.2f} s]".format(t2 - t1))

    labels_df = pd.DataFrame.from_dict(labels_list)
    # name should include resolution
    #labels_csv_name = "usgs_"+str(datetime.today())[:16].replace(':','.').replace('-','.').replace(' ','_') + ".csv"
    labels_df.to_csv(LABELS_FOLDER + labels_csv_name, index=False)

# Degrade images and save
if UPLOAD_LABEL_TO_DB:
    print("\nUploading labels CSV to the DB ...")
    t1 = time.time()
    #images_fullnames = ima.list_path_of_images_by_category(subfolder_base_res, category)
    #ima.degrade_images_and_save(images_fullnames, params, subfolder_size, category, images_usgs_col)
    t2 = time.time()
    print("[{:.2f} s]".format(t2 - t1))

print("\nDONE! [{:.2f} s]\n".format(t2 - t0))
