"""This module contains functions to test HW 1 in AE 498 Computational Systems Engineering.

"""

import numpy as np
import mlr as mlr


def load_data():
    """Function to load test data."""

    # load data
    fname = "transistor-gain-data.csv"
    fdata = np.loadtxt(fname, skiprows=1, delimiter=",")
    n, n_columns = np.shape(fdata)
    p = n_columns - 1
    X = np.concatenate((np.ones((n, 1)), fdata[:, 0:2]), axis=1)
    X_normalized = mlr.normalize_data(X)
    y = fdata[:, 2]

    return X_normalized, y, p


def test_model_fit_coefficients():
    """Function to test the model_fit function coefficients."""

    # load data
    X, y, _ = load_data()

    coefficients, _ = mlr.model_fit(X, y)
    coefficients_solution = [1242.31475505, 323.43473675, -54.77359489]

    np.testing.assert_allclose(coefficients, coefficients_solution)


def test_model_fit_y_predicted():
    """Function to the model_fit function y_predicted."""

    # load data
    X, y, _ = load_data()

    _, y_predicted = mlr.model_fit(X, y)
    y_predicted_solution = [
        973.65361318,
        1620.52308669,
        882.36428837,
        1529.23376187,
        1266.658575,
        1281.87346247,
        1205.79902512,
        928.00895078,
        1574.87842428,
        1297.08834993,
        1190.58413765,
        1251.44368753,
        1187.54116016,
        1305.34947699,
    ]

    np.testing.assert_allclose(y_predicted, y_predicted_solution)


def test_anova_p_value():
    """Function to test the anova function p-value."""

    # load data
    X, y, p = load_data()
    _, y_predicted = mlr.model_fit(X, y)

    p_value, _ = mlr.anova(y, y_predicted, p)
    p_value_solution = 4.741632197924487e-10

    np.testing.assert_allclose(p_value, p_value_solution)


def test_anova_f_statistic():
    """Function to test the anova function f-statistic."""

    # load data
    X, y, p = load_data()
    _, y_predicted = mlr.model_fit(X, y)

    _, f_statistic = mlr.anova(y, y_predicted, p)
    f_statistic_solution = 267.177

    np.testing.assert_allclose(f_statistic, f_statistic_solution)


def test_coefficient_tests_p_value():
    """Function to test the coefficient tests function p-values."""

    # load data
    X, y, _ = load_data()
    coefficients, y_predicted = mlr.model_fit(X, y)

    p_values, _ = mlr.coefficient_tests(X, coefficients, y, y_predicted)
    p_values_solution = [
        np.float64(5.656326763367771e-19),
        np.float64(1.349124579633935e-10),
        np.float64(0.0016204993349916484),
    ]

    np.testing.assert_allclose(p_values, p_values_solution)


def test_coefficient_tests_t_statistics():
    """Function to test the coefficient tests function t-statistics."""

    # load data
    X, y, _ = load_data()
    coefficients, y_predicted = mlr.model_fit(X, y)

    _, t_statistics = mlr.coefficient_tests(X, coefficients, y, y_predicted)
    t_statistics_solution = [
        np.float64(132.52099347689196),
        np.float64(22.729731459346127),
        np.float64(-4.14852225458658),
    ]

    np.testing.assert_allclose(t_statistics, t_statistics_solution)
