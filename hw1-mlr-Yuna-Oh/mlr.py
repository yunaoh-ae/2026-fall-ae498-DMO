"""This module contains functions to perform multiple linear regression for HW 1 in AE 498 Computational Systems Engineering.

"""

import numpy as np
import scipy.stats


def model_fit(X, y):
    """Function to fit a multiple linear regression model to a given dataset using ordinary least squares estimation.

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor. The first column should be all ones.
    y : array_like
        Output/response data (n elements), where each element is an observation.

    Returns
    -------
    coefficients : array_like
        Estimated model coefficients (1 + p elements).
    y_predicted : array_like
        Predicted responses for input data (n elements).

    """

    XT = X.T
    XT_X = np.dot(XT, X)
    XT_y = np.dot(XT, y)

    XT_X_inv = np.linalg.inv(XT_X)
    coefficients = np.dot(XT_X_inv, XT_y)

    y_predicted =  np.dot(X, coefficients)


    return coefficients, y_predicted



def anova(y, y_predicted, p):
    """Function to perform an ANOVA F-test for a multiple linear regression model.

    Parameters
    ----------
    y : array_like
        Output/response data (n elements), where each element is an observation.
    y_predicted : array_like
        Predicted responses for input data (n elements).
    p : int
        Number of predictors.

    Returns
    -------
    p_value : float
        P-value for model F-statistic.
    f_statistic : float
        F-statistic for model F-test.

    """
    
    n = len(y)
    y_mean = np.mean(y)

    ess = np.sum((y_predicted - y_mean) ** 2)
    rss = np.sum((y - y_predicted) ** 2)

    dof_ess = p
    dof_rss = n - p - 1

    f_statistic = (ess / dof_ess) / (rss / dof_rss)

    p_value = scipy.stats.f.sf(f_statistic, dof_ess, dof_rss)


    return p_value, f_statistic



def coefficient_tests(X, coefficients, y, y_predicted):
    """Function to perform hypothesis t-tests for coefficients of a multiple linear regression model.

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor. The first column should be all ones.
    coefficients : array_like
        Estimated model coefficients (1 + p elements).
    y : array_like
        Output/response data (n elements), where each element is an observation.
    y_predicted : array_like
        Predicted responses for input data (n elements).

    Returns
    -------
    p_values : list_like
        P-values for coefficient t-statistics.
    t_statistics : list_like
        T-statistics for coefficient t-tests.

    """

    n = X.shape[0]
    p = X.shape[1] - 1

    dof_rss = n - p - 1

    rss = np.sum((y - y_predicted) ** 2)
    var_s = rss / dof_rss

    XT = X.T
    XT_X = np.dot(XT, X)
    XT_X_inv = np.linalg.inv(XT_X)

    var_beta = var_s * np.diag(XT_X_inv)
    se_beta = np.sqrt(var_beta)

    t_statistics = coefficients / se_beta

    t_statistics_abs = np.abs(t_statistics)

    p_values = 2.0 * scipy.stats.t.sf(t_statistics_abs, dof_rss)


    return p_values, t_statistics



def normalize_data(X):
    """Function to normalize input data to [-1, 1].

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor. The first column should be all ones.

    Returns
    -------
    X_normalized : array_like
        Input data (n x (p + 1) dimensions) normalized to the range [-1, 1].

    """

    n, n_columns = np.shape(X)

    # normalize each column
    X_normalized = np.ones((n, n_columns))
    for column in range(n_columns):
        x_column = X[:, column]
        x_max = max(x_column)
        x_min = min(x_column)
        if x_max != x_min:
            x_column_normalized = [
                (x - (x_max + x_min) / 2) / ((x_max - x_min) / 2) for x in x_column
            ]
            X_normalized[:, column] = x_column_normalized

    return X_normalized
