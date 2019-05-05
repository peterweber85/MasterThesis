

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

def cross_validate_clf(X, y, clf, params):
    """
    Inputs:
        - X (ndarray): feature matrix, whole dataset
        - y (list or ndarray): target vector, whole dataset
        - clf (sklearn classifier)
        - params (dict): parameters for classifier, must contain field 'nfold'
    Outputs:
        - score: (list) 4-fold cross validation score computed using rf classifier
        - prediction: (list) of predicted sentiments with dimension of y
    """
    folds = params.pop('nfold', 3)
    clf = clf(**params)
    scores = cross_val_score(clf, X, y, cv=folds)
    prediction = cross_val_predict(clf, X, y, cv=folds)
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    return prediction, scores, score_mean, score_std


def zero_encoding(x):
    if x == 0: return 0
    if x == 1: return 0
    if x == 2: return 1


def two_encoding(x):
    if x == 0: return 0
    if x == 1: return 1
    if x == 2: return 1


def convert_encoding(y, encoding):
    y = [encoding(el) for el in y]
    return y


