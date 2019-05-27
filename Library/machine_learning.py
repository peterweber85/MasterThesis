

import numpy as np
import os

import deep_learning as dl


from sklearn.model_selection import cross_val_score, cross_val_predict


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


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX, binary = True):
    # make predictions
    yhats = [model.predict_classes(testX) for model in members]
    yhats = np.array(yhats)
    if len(yhats.shape) == 3:
        yhats = yhats[:,:,0]
    # average predictions of all models
    average = np.mean(yhats, axis=0)
    # return binary
    if binary:
        result = np.where(average >= 0.5, 1, 0)
    else:
        result = average
    return result


def load_ensemble_models(path, res, num_folds):

    folds = range(num_folds)
    models = []
    for fold in folds:
        fullpath = os.path.join(path, "model_res" + str(res) + "_fold" + str(fold))
        models.append(dl.load_keras_model(fullpath))
    return models