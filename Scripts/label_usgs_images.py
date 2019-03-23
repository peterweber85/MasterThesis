

#%%
from dotenv import load_dotenv
import os

from PIL import Image

import random
import numpy as np

#%%
# In parent directory, to get env variables
dotenv_path = ('../.env')
load_dotenv(dotenv_path)

#%%
BASE_FOLDER = os.getenv('USGS_FOLDER')
MFP_FOLDER = os.getenv('MFP_FOLDER')

#%%
filenames =  os.listdir(MFP_FOLDER + "processedIMG")
Image.open(MFP_FOLDER + "processedIMG/" + filenames[0]).show()


#%%
def label_images(filenames, labelled_images, num_images):
    unlabelled = list(set(filenames) - set(labelled_images))
    random.shuffle(unlabelled)
    for i in range(num_images):
        Image.open(MFP_FOLDER + "processedIMG/" + unlabelled[i]).show()
        label = input("Label:")

#%%
label_images(filenames, [], 1)


#%%
