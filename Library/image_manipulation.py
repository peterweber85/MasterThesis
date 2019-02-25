

# GENERAL:
from dotenv import load_dotenv
import os

# MAP & IMG:
from PIL import Image
import imageio

# Library
import db_connection as dbcon


# In file directory, to get google maps api key
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# In parent directory, to get google maps api key
dotenv_path = ('../.env')
load_dotenv(dotenv_path)


def load_images_from_gdrive(fnames, folder):
    """
    Returns list of dictionaries where every list entry has attributes
        'array': numpy array of the image
        'image': image file
        'fname': filename
    provided filenames and folder

    :param fnames: list of str
        filenames of the images to load from gdrive
    :param folder: str
        folder of images in gdrive
    :return:
    """
    images = []
    for fname in fnames:
        dict_ = {}
        dict_["array"] = imageio.imread(folder+fname)[:,:,:3]
        dict_["image"] = Image.open(folder+fname)
        dict_["fname"] = fname
        images.append(dict_)
    return images


def get_filenames_of_city(city, folder):
    """
    Return list of filenames that contain the name of the city specified by city,
    can also be a general name for an area
    :param city: str
    :param folder: str
        folder of images in gdrive
    :return:
    """
    fnames = []
    for fname in os.listdir(folder):
        if city in fname:
            fnames.append(fname)
    return fnames


def label_image(filename, label):
    """
    Give label to image in db that corresponds to filename=filename
    :param filename: str
    :param label: int
        one of 0,1,2,3,4 where 0 maximum natural and 4 maximum man-made
    :return:
    """
    db = dbcon.connect("../credentials/mlab_db.txt", "mfp")
    images_lib_col = db["images_lib"]

    query = {"filename": filename}
    new_val = {"$set": {"label": label}}
    images_lib_col.update_one(query, new_val)
    print("Image with filename", filename, "was labelled with", label)