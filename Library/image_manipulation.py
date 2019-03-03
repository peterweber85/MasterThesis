

# GENERAL:
from dotenv import load_dotenv
import os
import csv

# MAP & IMG:
from PIL import Image
import imageio
from IPython.display import display, clear_output

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


def get_filenames_of_city(city, folder, zoom = None):
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
        if not zoom is None:
            zoom_str = '_' + str(zoom) + '_'
            if city in fname and zoom_str in fname:
                fnames.append(fname)
        else:
            if city in fname:
                fnames.append(fname)
    return fnames


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


def add_labels_to_image_info(images_info_list):
    """
    Adds multi level and binary label to images_info_list. Precisely, it will create
    images_info_list['label_multi' + <name_initials>]
    images_info_list['label_binary' + <name_initials>]

    :param images_info_list: list of dict
        output of load_images_from_gdrive, i.e. every list element must contain key 'image'
    :return:
    """
    cnt = 0
    for image_info in images_info_list:
        display(image_info['image'])

        # Determine author of the label
        if cnt == 0:
            name = input("Your name? ")
            cnt += 1
        if name[:3].lower() == 'edu':
            label_multi = 'label_multi_er'
            label_binary = 'label_binary_er'
        elif name[:3].lower() == 'pet':
            label_multi = 'label_multi_pw'
            label_binary = 'label_binary_pw'
        else:
            label_multi = 'label_multi_other'
            label_binary = 'label_binary_other'

        # multi level label from 0 to 4
        image_info[label_multi] = int(input("\nProvide multiclass label: "))

        # binary label, is automatically = 1 if multi label > 0
        if image_info[label_multi] > 0:
            image_info[label_binary] = 1
        else:
            image_info[label_binary] = int(input("\nProvide binary label: "))
        # clear image output
        clear_output()

    return images_info_list


def save_labels_as_csv(images_info_list, output_folder, output_name, label_multi_name, label_binary_name):
    """
    Takes label info provided in images_info_list and writes a csv file with three columns
        - label_binary + <name>
        - label_multi + <name>
        - filename
    The csv file is saved in output_folder under output_name.

    :param images_info_list: list of dict
        output of add_labels_to_image_info
    :param output_folder: str
    :param output_name: str
    :param label_multi_name: str
    :param label_binary_name: str
    :return:
    """
    labels_multi = [images_info_list[i][label_multi_name] for i in range(len(images_info_list))]
    labels_binary = [images_info_list[i][label_binary_name] for i in range(len(images_info_list))]
    filenames = [images_info_list[i]["fname"] for i in range(len(images_info_list))]

    pd.DataFrame({
        label_multi_name: labels_multi,
        label_binary_name: labels_binary,
        'filename': filenames
    }).to_csv(output_folder + output_name, index=None, header=True)