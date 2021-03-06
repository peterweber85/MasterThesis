

# GENERAL:
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from scipy import misc

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

IMG_EXTENSIONS = ['.tif','.png','.jpg']


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


def reduce_image_quality(im, factor):
    """
    Reduce the number of pixels of an image by factor. factor should be a multiple of 2.

    Loaded libraries:
    - import numpy as np
    - from PIL import Image

    :param im: np.array or PIL.Image
    :param factor: int
        mulitpler of 2
    :return: np.array
    """
    # In case image is format PIL.Image convert to np.array
    if isinstance(im, Image.Image):
        im = np.array(im)
    # In case factor is smaller 1, return original image
    if factor <= 1:
        return im
    # Image dimensions
    length = im.shape[0]
    new_length = int(length / factor)
    channels = im.shape[2]
    # Use these indices from old image to construct new, downsized image
    indices = np.linspace(0 + int(factor / 2), length - int(factor / 2), new_length, dtype=int)
    # Initialize new, downsized image
    output = np.zeros((new_length, new_length, channels))
    for channel in range(channels):
        output[:, :, channel] = np.array([[im[row, col, channel] for col in indices] for row in indices])
    output = output.astype('uint8')

    return output


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

        print(i+1, "/", num_images)
        print(images_info['fname'][i])
        print()

        print(labeling_info_multi)
        print(labeling_info_binary)

        label_multi_i = int(input("\nProvide multiclass label: "))
        images_info[label_multi].append(label_multi_i)

        # binary label, is automatically = 1 if multi label > 0
        if label_multi_i > 0:
            label_binary_i = 1
            images_info[label_binary].append(label_binary_i)
        else:
            #label_binary_i = int(input("\nProvide binary label: "))
            label_binary_i = 0
            images_info[label_binary].append(label_binary_i)

        labeling_df = labeling_df.append({
            'filename': images_info['fname'][i],
            label_multi: label_multi_i,
            label_binary: label_binary_i
        }, ignore_index=True)
        labeling_df.to_csv(output_folder + output_name, index=None, header=True)

        # clear image output
        clear_output()

    print("DONE!")
    print("Output file:", output_name)

    return images_info

def correct_tile(img):
    x_correction = img.tile[0][1][2]
    y_correction = img.tile[0][1][3]

    img.tile = [e for e in img.tile if e[1][2] < img.size[0] and e[1][3] < img.size[1]]

    array = np.array(img)[:-x_correction, :-y_correction, :3]
    return array

def load_image_as_rgb_array(file):
    try:
        array = misc.imread(file, mode='RGB')
        if array.size <= 1:
            print("File with filename", file, "couldn't be processed with SCIPY, correcting tile...")
            img = array.all()
            array = correct_tile(img)
        return array
    except:
        print("File with filename", file, "couldn't be processed with SCIPY, correcting tile...")
        img = Image.open(file)
        array = correct_tile(img)
        return array


def list_path_of_images_by_category_and_label(image_folder, category, label = None, extensions=IMG_EXTENSIONS):
    paths = []
    if label is None:
        for filename in os.listdir(os.path.join(image_folder, category)):
            if os.path.splitext(filename)[1] in extensions:
                paths.append(os.path.join(image_folder, category, filename))
    else:
        for filename in os.listdir(os.path.join(image_folder, category, "label_" + str(label))):
            if os.path.splitext(filename)[1] in extensions:
                paths.append(os.path.join(image_folder, category, "label_" + str(label), filename))
    return paths

def create_csv_with_labels_by_category_usgs(folder_base_res, category, labels):
    dict_labels = {}
    for label in labels:
        paths = list_path_of_images_by_category_and_label(folder_base_res, category, label)
        for path in paths:
            filename = path.split("/")[-1]
            dict_labels[filename] = label
    df_labels = pd.DataFrame(list(dict_labels.items()), columns = ['filename', 'label'])
    df_labels.to_csv(folder_base_res + "labels-" + category + ".csv", index=False)

def create_directory(path):
    try:
        os.mkdir(path)
        print("Directory", path, "was created!")
    except:
        print("Directory", path, "already exists!")


#%% Degrade images and save
def degrade_images_and_save(paths, params, root_folder, category, db_collection, downsample = Image.LANCZOS):

    size = params['size']
    res = params['res']
    res_degraded = params['res_degr']

    if isinstance(res_degraded, int) or isinstance(res_degraded, float):
        res_degraded = [res_degraded]

    for factor in res_degraded: # factor: degraded resolution in meters
        new_size = round(size/(factor/res))
        new_folder = root_folder + "usgs_" + str(size) + "_" + str(factor) + "m/"
        create_directory(new_folder)
        new_folder = new_folder + category + "/"
        create_directory(new_folder)
        for path in paths:
            imarray = load_image_as_rgb_array(path)
            imresize = Image.fromarray(imarray).resize((new_size, new_size), resample = downsample)
            filename = path.split("/")[-1]
            new_filename = filename.replace("_res" + str(res) + "m", "_res" + str(factor) + "m")
            output_path = new_folder + new_filename
            imresize.save(output_path)
            gist_vector = gist.extract(np.array(imresize), nblocks=1, orientations_per_scale=(8, 8, 4)).tolist()
            result = db_collection.update(
                {"filename": filename},
                {"$set": {"gist_"+str(factor).replace(".", "_"): gist_vector}}
            )
            if result["nModified"] == 0:
                print(filename, "GIST not uploaded to the DB!")


def move_folder_content(source_folder, target_folder, extensions=IMG_EXTENSIONS):

    # create parent of target_folder
    create_directory("/".join(target_folder.split("/")[:-1]))
    # create target_folder
    create_directory(target_folder)

    for filename in os.listdir(source_folder):
        if os.path.splitext(filename)[1] in extensions:
            os.rename(
                os.path.join(source_folder, filename),
                os.path.join(target_folder, filename)
            )
    return

def degrade_image(imarray, size_degraded, keep_size = False):
    """
    Degrades image to size_degraded, and optionally keeps the original size in pixels
    :param img: np.array or PIL.Image
    :param size_degraded: 2-tuple
    :return: np.array
    """
    if isinstance(imarray, np.ndarray):
        img = Image.fromarray(imarray)
    else:
        img = imarray

    img = img.resize((size_degraded[0], size_degraded[1]), resample = Image.LANCZOS)
    if keep_size:
        dim = np.array(img).shape
        img = img.resize((dim[0], dim[1]))
    return np.array(img)


def load_images_into_df(paths, label = None):
    """
    Returns dataframe with 2 columns: filename and image array
    :param paths: list of strings
        with full file path
    :return:
    """

    filenames = [fname.split("/")[-1] for fname in paths]
    images = [load_image_as_rgb_array(fname) for fname in paths]
    df = pd.DataFrame({
        'filename': filenames,
        'image': images
    })
    if not label is None:
        df.label = label

    return df


def load_degraded_images_into_df(df_images, sizes, label = False):
    """
    :param df_images: df
        base resolution images, needs to have columns 'image' containing
        numpy array or PIL.Image of images
    :param sizes: dict
        keys: resolution per pixel (as a string)
        values: number of pixels of degraded image (as a tuple)
    :return:
    """
    if df_images.shape[0] < 2:
        print("DataFrame must have at least 2 rows")
        return

    if label:
        df = pd.DataFrame(columns=['filename', 'image', 'resolution', 'label', 'category'])
    else:
        df = pd.DataFrame(columns=['filename', 'image', 'resolution', 'category'])

    for resolution in sizes.keys():
        df_res = pd.DataFrame({
            'filename': df_images.filename,
            'image': [degrade_image(image, sizes[resolution])
                      for image in df_images.image],
            'resolution': resolution,
            'category': df_images.category
        })
        if label:
            df_res['label'] = df_images.label
        df = df.append(df_res)
    return df.reset_index(drop = True)


def load_images_into_df_by_category_and_label(base_folder, category, label=None, base_res=0.3):
    """
    Preprocessing function that creates dataframe with 3 optional 4 columns
    - filename
    - image
    - resolution
    - label (optional)
    - category

    :param base_folder: str
    :param category: str
    :param label: int
    :param base_res: float
    :return:
    """
    fnames = list_path_of_images_by_category_and_label(base_folder, category, label)
    df_images = load_images_into_df(fnames, label)
    df_images["resolution"] = base_res
    if not label is None:
        df_images["label"] = label
    df_images["category"] = category

    return df_images

