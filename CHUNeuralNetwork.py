from sklearn.base import TransformerMixin
import numpy as np
from math import sqrt
from scipy.integrate import odeint


# %% defining external equaltions


def rank_finder(hidden_neurons):
    """find and return the indices of the 1st and 7th neurons by activation.

    Argsort the neurons, the select the 1rs and 7th ones from each sample.

    arguments
    ---------
    hidden_neurons
        the neurons to be sorted
    return
    ------
    ndarray, shape (2, no. of samples in the batch)
    """
    sort = np.argsort(hidden_neurons)
    # sorts along last axis by default
    first_neurons = sort[:, -1]
    seventh_neurons = sort[:, -7]
    return (first_neurons, seventh_neurons)


def product(weight_matrix, batch, p):
    """."""
    coefficients = np.abs(weight_matrix) ** p
    return np.einsum("ik,jk,jk->ij", batch, weight_matrix, coefficients)

