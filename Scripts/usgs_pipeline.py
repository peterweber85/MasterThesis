
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

#%% parameters
PROCESS_RAW_IMAGES = False

GDRIVE_FOLDER = os.getenv('GDRIVE_FOLDER')
RAW_IMAGE_FOLDER = GDRIVE_FOLDER + 'images_usgs/'
IMAGE_FOLDER = GDRIVE_FOLDER + 'MFP - Satellogic/images/usgs/'
CATEGORIES = ['agriculture', 'city', 'forest-woodland', 'semidesert', 'shrubland-grassland']

#%% Helpers
def list_path_of_images_by_category(image_folder, category):
    filenames = os.listdir(image_folder + category)
    paths = [image_folder + category + '/' + filename for filename in filenames if filename.startswith("m")]
    return paths

#%%
def get_image_grid(imarray, size=512):
    dim = imarray.shape[:2]

    # Number of images along every axis
    num_x = int((dim[1] - size) / size) + 1
    num_y = int((dim[0] - size) / size) + 1

    # Center images
    x_move_by = int((dim[1] - num_x * size) / 2)
    y_move_by = int((dim[0] - num_y * size) / 2)

    # create dictionary of grid points
    grid = dict()
    for ix in range(num_x):
        for iy in range(num_y):
            grid[(iy, ix)] = (size * iy + y_move_by, size * ix + x_move_by)
    return grid


def get_cropped_images(imarray, grid):
    x_coord = [x[0] for x in list(grid.values())]
    size = x_coord[1] - x_coord[0]

    output = dict()
    for coord in grid.values():
        img = imarray[coord[0]:coord[0] + size, coord[1]:coord[1] + size]
        output[coord] = img

    return output


def main(raw_image_filename):
    # 1. CROP image
    # 2. Generate DOWNGRADE images
    # 3. Compute GISTS
    return


#%%
if __name__ == '__main__':
    # %%
    start_point = 3000
    end_point = 3800
    size = 256

    shrubland = list_path_of_images_by_category(RAW_IMAGE_FOLDER, 'shrubland-grassland')


    imarray = ima.load_image_as_rgb_array(shrubland[0])
    grid = get_image_grid(imarray, size)
    imcropped = get_cropped_images(imarray, grid)
    imcropped


#%%

#%%