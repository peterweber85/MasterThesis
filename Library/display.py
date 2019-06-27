
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import image_manipulation as ima

def parse_wrong_class_str(string):
    """
    When saving the results dataframes and loading them, pandas converts
    everything to string. So a list inside a field is converted to string.

    This function parses a string that was initially a list, and converts
    it back to list.

    :param string:
    :return:
    """
    if isinstance(string, str):
        output = string.replace('[','').replace(']','').split(",")
        output = [int(idx) for idx in output]
    else:
        output = string
    return output


def get_misclf_or_correctclf_images(X, y, df_results, res, n=None, misclf=True):
    """
    Returns sample (n) of misclfassified (default) or correctly classified images
    :param X: np.array
        of images
    :param y: list or np.array
        labels
    :param df_results: df
        saved results df from Colab
    :param res: float
    :param n: int
        number of images to return
    :param misclf: bool
    :return: tuple
    """

    # Get wrong_class column from df
    idx_series = df_results[df_results.resolution == res].wrong_class

    # Convert to one list with all indices
    indcs = []
    for item in idx_series:
        item = parse_wrong_class_str(item)
        indcs = indcs + item

    # Get misclassified or correctly classified indices
    if not misclf:
        indcs = list(set([i for i in range(X.shape[0])]) - set(indcs))

    # Sample
    if not n is None:
        indcs = random.sample(indcs, k=n)

    # subset X, y with indices
    if len(indcs) > 0:
        X = X[indcs]
        y = y[indcs]

    return X, y, indcs


def plot_misclf_or_correctclf_images(X, y = None, n = None, columns=4, misclf=None, degrade = None, show_real_label = True):
    """

    :param X:
    :param y:
    :param n:
    :param columns:
    :param misclf:
    :param degrade: tuple
    :return:
    """
    if n is None: n = X.shape[0]
    last_line = n % columns

    if not degrade is None:
        X = np.array([ima.degrade_image(X[i], degrade) for i in range(X.shape[0])])

    fig = plt.figure(figsize=(15, 4 * n / columns))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(n):
        ax = fig.add_subplot(int(n / columns), columns, np.max([i + 1 - last_line, 1]))
        ax.imshow(X[i, :, :, :])

        if not y is None:
            if misclf and show_real_label:
                ax.axes.set_title('Real Label: ' + str(y[i]) + ', Classified as ' + str(1 - y[i]))
            elif not misclf and show_real_label:
                ax.axes.set_title('Real Label: ' + str(y[i]) + ', Classified as ' + str(y[i]))
            elif not show_real_label:
                ax.axes.set_title('\nClassified as ' + str(y[i]))
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    fig.tight_layout()
    plt.show()
    return fig





def plot_image_array(X, n = None, columns = 4, fname_save = None, resolutions = None):

    if isinstance(X, dict):
        X = [X[round(res, 1)] for res in X.keys()]

    if n is None:
        n = len(X)
    last_line = n % columns

    fig = plt.figure(figsize=(15, 4 * n / columns))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(n):
        ax = fig.add_subplot(int(n / columns), columns, np.max([i + 1 - last_line, 1]))
        ax.imshow(X[i])
        ax.grid()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

        if not resolutions is None:
            ax.set_title("Resolution per pixel: " + str(resolutions[i]) + "m", fontsize=14)
    fig.tight_layout()

    if not fname_save is None:
        plt.savefig(fname_save)


def get_image_statistics(base_folder, categories, labels):
    df_images = pd.DataFrame(columns=['filename', 'image', 'resolution', 'label', 'category'])
    for category in categories:
        for label in labels:
            df_images = df_images.append(ima.load_images_into_df_by_category_and_label(base_folder,
                                                                                       category,
                                                                                       label)
                                         )
    df_counts_by_category = df_images.groupby(['category', 'label']).size().reset_index(name='counts')

    imstats = df_counts_by_category.pivot(index='category', columns='label', values='counts')
    imstats.columns = ['label_' + str(x) for x in imstats.columns]
    imstats = imstats.reset_index()

    return imstats, df_images


