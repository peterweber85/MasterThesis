

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


def get_image_filenames(img_folder, img_ext = ['png']):
    """
    Returns list of all image filenames in 'img_folder'
    :param img_folder:
    :param img_ext:
    :return:
    """
    images_files = [file for file in os.listdir(img_folder) if any(file.endswith(ext) for ext in img_ext)]
    print("Existing images files:", len(images_files))
    return images_files

def get_metadata_filenames(db_collection):
    """
    Returns filenames of metadata entries in db_collection
    :param db_collection:
    :return:
    """
    images_metadata = [img_metadata["filename"] for img_metadata in db_collection.find({})]
    print("Existing images metadata:", len(images_metadata))
    return images_metadata


def get_discrepancies_between_metadata_and_images(images_files, images_metadata):
    """
    Returns both
        - missing metadata where existing image and
        - missing image where existing metadata
    :param images_files: list
        of filenames from shared folder
    :param images_metadata: list
        of filenames of metadata in db
    :return:
    """
    missing_metadata = list(set(images_files) - set(images_metadata))
    print("Missing metadata for " + str(len(missing_metadata)) + " files:")
    for name in missing_metadata:
        print(" " + name)

    missing_files = list(set(images_metadata) - set(images_files))
    print("Missing files for " + str(len(missing_files)) + " metadata:")
    for name in missing_files:
        print(" " + name)

    return missing_metadata, missing_files