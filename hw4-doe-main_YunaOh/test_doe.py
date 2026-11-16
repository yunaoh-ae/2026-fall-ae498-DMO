"""This module contains test functions for this assignment.

"""

import numpy as np
import pandas as pd
import doe


def load_data(fname):
    """Function to load test data.

    """

    # load data
    df = pd.read_csv(fname)
    _, n_replicates = df.shape
    responses = df.to_numpy()

    return responses, n_replicates


def test_effects():
    """Function to test the effects function."""

    # load data 1
    responses, n_replicates = load_data('bacterial-growth.csv')
    effects = doe.effects(responses, n_replicates)
    effects_solution = [-1.25, 9.916667, -3.916667]

    np.testing.assert_allclose(effects, effects_solution, rtol=1e-05)

    # load data 2
    responses, n_replicates = load_data('catalyst-reactant.csv')
    effects = doe.effects(responses, n_replicates)
    effects_solution = [8.333333, -5, 1.666667]

    np.testing.assert_allclose(effects, effects_solution, rtol=1e-05)
