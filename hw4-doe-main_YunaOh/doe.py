"""This module contains functions for design of experiments analysis for HW 4 in AE 498 Data-Driven Modeling and Optimization."""
import numpy as np
import matplotlib.pyplot as plt


def effects(responses, n_replicates):
    """Function to calculate main and interaction effects for a 2^2 factorial experiment.

    Parameters
    ----------
    responses : array_like
        Input response data (4 x n_replications dimensions), where each row is an observed response for a condition (ordered in the standard order) and each column is a replicate.
        For example, responses[0,1] = response from replicate 1 for condition (1), responses[1,1] = response from replicate 1 for condition a.
    n_replicates : int
        Number of replications in the experiment data.

    Returns
    -------
    effects : array_like
        Calculated effects (3 x 1 dimensions) for A, B, and AB.

    Notes
    -----
    Standard order of experiment conditions (order of rows in responses) = (1), a, b, ab
    """

    responses = np.asarray(responses)

    sum_1, sum_A, sum_B, sum_AB = np.sum(responses, axis=1)
    
    A_effect = (sum_A + sum_AB - sum_1 - sum_B) / (2*n_replicates)
    B_effect = (sum_B + sum_AB - sum_1 - sum_A) / (2*n_replicates)
    AB_effect = (sum_1 + sum_AB - sum_A - sum_B) / (2*n_replicates)

    effects = np.array([A_effect, B_effect, AB_effect])

    return effects


def plot_effects(responses, n_replicates):
    """Function to plot effects for a 2^2 factorial experiment.

    Parameters
    ----------
    responses : array_like
        Input response data (4 x n_replications dimensions), where each row is an observed response for a condition (ordered in the standard order) and each column is a replicate.
        For example, responses[0,1] = response from replicate 1 for condition (1), responses[1,1] = response from replicate 1 for condition a.
    n_replicates : int
        Number of replications in the experiment data.

    Returns
    -------
    out : list
        List of artists added.

    Notes
    -----
    This function structure is based on matplotlib's suggested signature function for returning a plot:
    https://matplotlib.org/3.5.0/tutorials/introductory/usage.html#the-object-oriented-interface-and-the-pyplot-interface
    """

    y_1, y_A, y_B, y_AB = np.mean(responses, axis=1)

    fig, ax = plt.subplots()

    line_Bp = ax.plot([-1, 1], [y_B, y_AB], 'o--', label='B+')
    line_Bm = ax.plot([-1, 1], [y_1, y_A], 'o--', label='B-')

    ax.set_xticks([-1, 1])
    ax.set_xticklabels(['Low, -', 'High, +'])
    ax.set_xlabel('Factor A')
    ax.set_ylabel('Response')
    ax.legend()

    out = line_Bp + line_Bm

    return out


def plot_tornado(responses, n_replicates):
    """Function to make a tornado plot for a 2^2 factorial experiment.

    Parameters
    ----------
    responses : array_like
        Input response data (4 x n_replications dimensions), where each row is an observed response for a condition (ordered in the standard order) and each column is a replicate.
        For example, responses[0,1] = response from replicate 1 for condition (1), responses[1,1] = response from replicate 1 for condition a.
    n_replicates : int
        Number of replications in the experiment data.

    Returns
    -------
    out : list
        List of artists added.

    Notes
    -----
    This function structure is based on matplotlib's suggested signature function for returning a plot:
    https://matplotlib.org/3.5.0/tutorials/introductory/usage.html#the-object-oriented-interface-and-the-pyplot-interface
    """

    effect_result = effects(responses, n_replicates)

    fig, ax = plt.subplots()

    bars = ax.barh(['A', 'B', 'AB'], effect_result)

    ax.invert_yaxis()
    ax.axvline(x=0, color='black')
    ax.set_xlabel('Average Effect')
    ax.set_ylabel('Factor of Interaction')

    out = list(bars)

    return out
