
import numpy as np
import pandas as pd


from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras.models import model_from_json

import image_manipulation as ima


def get_activations(img, base_model, activation_name, print_ = False):
    """
    Returns ReLUs of layer specified by parameter activation

    :param img: np.array
    :param base_model: Keras Model
    :param activation_name: str
    :return:
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)
    if print_:
        print("Shape of input:", img.shape)

    if isinstance(activation_name, int):
        model = Model(inputs=base_model.input,
                      outputs=base_model.get_layer('activation_' + str(activation_name)).output)
    else:
        model = Model(inputs=base_model.input,
                      outputs=base_model.get_layer(activation_name).output)

    activation = model.predict(img)
    if print_:
        print("Shape of output:", activation.shape)

    return activation


def preprocess_activations(X, model, degrade = None, activation_name='conv5_block3_out'):

    if not degrade is None:
        print("--- degrading images ---")
        X = np.array([ima.degrade_image(X[i], degrade) for i in range(X.shape[0])])

    print("--- getting activations ---")
    X_act = get_activations(X, model, activation_name)

    dim = X_act.shape
    X_act = preprocess_input(X_act)
    X_act_flat = np.reshape(X_act, (dim[0], dim[1] * dim[2] * dim[3]))

    return X_act_flat


def generate_X_y_from_df(df_images, resolution = None):

    if not resolution is None:
        mask = df_images.resolution == resolution
        images_array = np.stack(df_images[mask].image)
        labels = df_images[mask]["label"]
        filenames = df_images[mask]["filename"]
        categories = df_images[mask]["category"]
    else:
        images_array = np.stack(df_images.image)
        labels = df_images["label"]
        filenames = df_images["filename"]
        categories = df_images["category"]

    print("Shape of image array is:", images_array.shape)
    return images_array, \
           np.array(labels, dtype=pd.Series), \
           np.array(filenames, dtype=pd.Series), \
           np.array(categories, dtype=pd.Series)


def load_keras_model(path, X = None, y = None):
    # load json and create model
    json_file = open(path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path + ".h5")
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    if not X is None:
        score = loaded_model.evaluate(X, y, verbose=0)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
        return loaded_model, score[1]
    return loaded_model