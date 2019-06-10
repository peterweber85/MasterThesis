
import random
import numpy as np
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
                ax.axes.set_title('Real Label: ' + str(y[i]) + '\nClassified as ' + str(1 - y[i]))
            elif not misclf and show_real_label:
                ax.axes.set_title('Real Label: ' + str(y[i]) + '\nClassified as ' + str(y[i]))
            elif not show_real_label:
                ax.axes.set_title('\nClassified as ' + str(y[i]))
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    fig.tight_layout()