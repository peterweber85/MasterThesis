
#%%
from dotenv import load_dotenv
import os

from PIL import Image

import numpy as np

#%%
# In parent directory, to get env variables
dotenv_path = ('../.env')
load_dotenv(dotenv_path)

#%%
BASE_FOLDER = os.getenv('USGS_FOLDER')
MFP_FOLDER = os.getenv('MFP_FOLDER')

#%%
def list_usgs_img_files(base_folder):
    # list all folders
    folders = os.listdir(base_folder)
    # exclude hidden folders
    folders = [folder for folder in folders if folder[0] != "."]
    # append all .tif images to output
    output = []
    filenames = []
    for folder in folders:
        folder = base_folder + folder
        files = os.listdir(folder)
        for file in files:
            if file.endswith(".tif"):
                output.append(folder + "/" + file)
                filenames.append(file)
    return output, filenames
def load_image_into_array(file):
    img = Image.open(file)
    array = np.array(img)
    return array
def get_image_grid(imarray, size=512):
    dim = imarray.shape[:2]

    num_x = int((dim[1] - size)/size) + 1
    num_y = int((dim[0] - size)/size) + 1

    grid = dict()
    for ix in range(num_x):
        for iy in range(num_y):
            grid[(iy, ix)] = (size * iy, size * ix)
    return grid
def get_cropped_images(imarray, grid):
    x_unique = list(set([x[0] for x in grid.values()]))
    size = x_unique[1] - x_unique[0]

    images = []
    coordinates = []
    for coord in grid.values():
        img = imarray[coord[0]:coord[0] + size, coord[1]:coord[1] + size, :4]
        images.append(img)
        coordinates.append(coord)
    return images, coordinates

#%%
full_filenames, filenames = list_usgs_img_files(BASE_FOLDER)
print("Number of images to process", len(filenames))

#%%
cnt = 1
for full_fname, fname in zip(full_filenames, filenames):
    print("Processing image", cnt)
    cnt += 1
    imarray = load_image_into_array(full_fname)
    grid = get_image_grid(imarray)
    images_grid, coordinates = get_cropped_images(imarray, grid)
    for image, coord in zip(images_grid, coordinates):
        fname_to_save = fname.split(".")[0] + "_x" + str(coord[0]) + "_y" + str(coord[1])
        Image.fromarray(image).save(MFP_FOLDER + 'processedIMG/' + fname_to_save + ".png")


#%%