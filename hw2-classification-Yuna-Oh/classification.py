"""This module contains functions to fit classification models for HW 2 in AE 498 Computational Systems Engineering.

"""

import numpy as np


def fit_knn(X, y, n_neighbors):
    """Function to fit a KNN classifier to a given dataset.

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor.
    y : array_like
        Output/response data (n elements), where each element is an observation.
    n_neighbors : int
        Number of k neighbors to use for each prediction.

    Returns
    -------
    y_predicted : array_like
        Predicted responses for input data (n elements).
    error : float
        Error rate for the input data, calculated as 1/n_obs * number of incorrect predictions.

    Notes
    -----
    The test functions assume the k nearest neighbors of observation x include x.

    """

    X = np.asarray(X)
    y = np.asarray(y)

    n_obs = X.shape[0]
    y_array = np.empty(n_obs, dtype=y.dtype)

    for i in range(n_obs):

        d = np.sqrt(np.sum((X - X[i]) **2, axis=1))

        find_k_idx = np.argsort(d)
        find_k_idx = find_k_idx[:n_neighbors]

        find_k_val = y[find_k_idx]

        max_pr = -1000
        class_pr = None
        for j in np.unique(y):

            pr = np.mean(find_k_val == j)

            if pr > max_pr:
                max_pr = pr
                class_pr = j

        y_array[i] = class_pr
    
    error = np.mean(y_array != y)
    y_predicted = y_array.tolist()

    return error, y_predicted

