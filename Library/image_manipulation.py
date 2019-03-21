

# GENERAL:
from dotenv import load_dotenv
import os
import pandas as pd

# MAP & IMG:
from PIL import Image, ImageDraw
import imageio
from IPython.display import display, clear_output
import gist

# Library
import db_connection as dbcon


# In file directory, to get google maps api key
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# In parent directory, to get google maps api key
dotenv_path = ('../.env')
load_dotenv(dotenv_path)


def load_images_from_gdrive(fnames, folder, return_list = False):
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
    if return_list:
        images = []
        for fname in fnames:
            dict_ = {}
            dict_["array"] = imageio.imread(folder+fname)[:,:,:3]
            dict_["image"] = Image.open(folder+fname)
            dict_["fname"] = fname
            images.append(dict_)
    else:
        images = {}
        images["array"] = []
        images["image"] = []
        images["fname"] = []
        for fname in fnames:
            images["array"].append(imageio.imread(folder + fname)[:, :, :3])
            images["image"].append(Image.open(folder + fname))
            images["fname"].append(fname)
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


def get_metadata_filenames(db_collection, query={}):
    """
    Returns filenames of metadata entries in db_collection
    :param db_collection:
    :return:
    """
    images_metadata = [img_metadata["filename"] for img_metadata in db_collection.find(query)]
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


def delete_images_files(folder, filenames):
    """
    Deletes image files from local folder
    :param folder: str
    :param filenames: list of str
    :return: number of files deleted
    """
    deleted_count = 0
    for filename in filenames:
        file_path = folder + filename
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted_count += 1
    return deleted_count


def gist_calculate_and_load(filenames, folder, db_collection):
    """
    Calculates gist vectors of the images and uploads them to the DB
    :param filenames: list of str
    :param folder: str
    :param db_collection: mongodb collection object
    :return: number of DB documents updated and computed gist vectors
    """

    images_files = load_images_from_gdrive(filenames, folder, return_list=True)

    gist_uploaded = 0
    gist_vectors = []
    for image in images_files:
        gist_vector = gist.extract(image["array"])
        gist_vectors.append(gist_vector)
        result = db_collection.update(
            {"filename": image["fname"]},
            {"$set": {"gist": gist_vector.tolist()}}
        )
        gist_uploaded += result["nModified"]

    return gist_uploaded, gist_vectors


def add_labels_to_image_info(images_info):
    """
    Adds multi level and binary label to images_info_list. Precisely, it will create
    images_info['label_multi' + <name_initials>]
    images_info'label_binary' + <name_initials>]

    :param images_info: dict of lists
        output of load_images_from_gdrive, i.e. every list element must contain key 'image'
    :return:
    """
    cnt = 0
    for i in range(len(images_info['fname'])):
        display(images_info['image'][i])

        # Determine author of the label
        if cnt == 0:
            name = input("Your name? ")

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
        if cnt == 0:
            images_info[label_multi] = []
            images_info[label_binary] = []
            cnt += 1
        images_info[label_multi].append(int(input("\nProvide multiclass label: ")))

        # binary label, is automatically = 1 if multi label > 0
        if images_info[label_multi][-1] > 0:
            images_info[label_binary].append(1)
        else:
            images_info[label_binary].append(int(input("\nProvide binary label: ")))
        # clear image output
        clear_output()

    return images_info


def save_labels_as_csv(images_info, output_folder, output_name, label_multi_name, label_binary_name):
    """
    Takes label info provided in images_info_list and writes a csv file with three columns
        - label_binary + <name>
        - label_multi + <name>
        - filename
    The csv file is saved in output_folder under output_name.

    :param images_info_list: dict of lists
        output of add_labels_to_image_info
    :param output_folder: str
    :param output_name: str
    :param label_multi_name: str
    :param label_binary_name: str
    :return:
    """
    pd.DataFrame({
        label_multi_name: images_info[label_multi_name],
        label_binary_name: images_info[label_binary_name],
        'filename': images_info['fname']
    }).to_csv(output_folder + output_name, index=None, header=True)


def add_labels_and_save_csv(images_info, output_folder, output_name):
    """
    Adds multi level and binary label to images_info_list. Precisely, it will create
    images_info['label_multi' + <name_initials>]
    images_info'label_binary' + <name_initials>]

    :param images_info: dict of lists
        output of load_images_from_gdrive, i.e. every list element must contain key 'image'
    :param output_folder: str
    :param output_name: str
    :return:
    """

    #labeling_info_multi = "Multiclass: 0 = 0-20%, 1 = 20-40%, 2 = 40-60%, 3 = 60-80%, 4 = 80-100%"
    labeling_info_multi = "Multiclass: 0-9 cells with human impact"
    labeling_info_binary = "Binary:     0 = no human impact, 1 = human impact"

    name = input("Your name? ")

    if name[:3].lower() == 'edu':
        label_multi = 'label_multi_er'
        label_binary = 'label_binary_er'
    elif name[:3].lower() == 'pet':
        label_multi = 'label_multi_pw'
        label_binary = 'label_binary_pw'
    else:
        label_multi = 'label_multi_other'
        label_binary = 'label_binary_other'

    # TO-DO: input for changing default label
    print("Labels: " + label_multi + ", " + label_binary )

    images_info[label_multi] = []
    images_info[label_binary] = []
    labeling_df = pd.DataFrame(index=[], columns=[label_multi, label_binary, 'filename'])

    num_images = len(images_info['fname'])
    for i in range(num_images):

        # load image and overlay 3x3 grid
        im = images_info['image'][i]
        width, height = im.size
        draw = ImageDraw.Draw(im)
        draw.line([(0, height / 3), (width, height / 3)], fill=1000)
        draw.line([(0, 2 * height / 3), (width, 2 * height / 3)], fill=1000)
        draw.line([(width / 3, 0), (width / 3, height)], fill=1000)
        draw.line([(2 * width / 3, 0), (2 * width / 3, height)], fill=1000)
        display(im)

        print("image", i+1, "of", num_images)
        print(labeling_info_multi)
        print(labeling_info_binary)

        label_multi_i = int(input("\nProvide multiclass label: "))
        images_info[label_multi].append(label_multi_i)

        # binary label, is automatically = 1 if multi label > 0
        if label_multi_i > 0:
            label_binary_i = 1
            images_info[label_binary].append(label_binary_i)
        else:
            label_binary_i = int(input("\nProvide binary label: "))
            images_info[label_binary].append(label_binary_i)

        labeling_df = labeling_df.append({
            'filename': images_info['fname'][i],
            label_multi: label_multi_i,
            label_binary: label_binary_i
        }, ignore_index=True)
        labeling_df.to_csv(output_folder + output_name, index=None, header=True)

        # clear image output
        clear_output()

    print("Output file:", output_name)

    return images_info


