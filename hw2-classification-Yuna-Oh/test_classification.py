"""This module contains functions to test HW 1 in AE 498 Computational Systems Engineering.

"""

import numpy as np
from ucimlrepo import fetch_ucirepo
import classification as cf


def load_data():
    """Function to load test data.
    
    The data comes from https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks.
    """

    # load data
    connectionist_bench_sonar_mines_vs_rocks = fetch_ucirepo(id=151)
    X_df = connectionist_bench_sonar_mines_vs_rocks.data.features
    y_df = connectionist_bench_sonar_mines_vs_rocks.data.targets
    X = X_df.to_numpy()
    y = y_df.to_numpy()[:,0]

    return X, y


def test_fit_knn_error():
    """Function to test the fit_knn function error."""

    # load data
    X, y = load_data()

    n_neighbors = 5
    error, _ = cf.fit_knn(X, y, n_neighbors)
    error_solution = 0.1346153846153846

    np.testing.assert_allclose(error, error_solution)


def test_fit_knn_y_predictions():
    """Function to test the fit_knn function y_predictions."""

    # load data
    X, y = load_data()

    n_neighbors = 5
    _, y_predictions = cf.fit_knn(X, y, n_neighbors)
    y_predictions_solution = ['M', 'M', 'R', 'R', 'R', 'M', 'R', 'M', 'R', 'R']

    assert(y_predictions[0:10] == y_predictions_solution)
