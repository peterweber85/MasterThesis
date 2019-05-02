
import numpy as np


from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.preprocessing import image



def get_activations(img, base_model, activation = 1):
    """
    Returns ReLUs of layer specified by parameter activation

    :param img: np.array
    :param base_model: Keras Model
    :param activation_: int
    :return:
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)
    print("Shape of input:", img.shape)

    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('activation_' + str(activation)).output)

    activation = model.predict(img)
    print("Shape of output:", activation.shape)
    return activation


def generate_X_y_from_df(df_images, resolution):
    mask = df_images.resolution == resolution

    images_array = np.stack(df_images[mask].image)
    labels = df_images[mask].label

    print("Shape of image array is:", images_array.shape)
    return images_array, labels