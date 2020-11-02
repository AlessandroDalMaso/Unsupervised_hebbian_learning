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
    first_indexes = sort[:, -1]
    seventh_indexes = sort[:, -7]
    return (first_indexes, seventh_indexes)


def product(weight_matrix, batch, p):
    """Multiply the inputs by the synapses weights.

    define coefficients, the multiply weight_matrix, coefficients, and the data
    in a single operation.

    parameters
    ----------
    weight_matrix
        the matrix of the synapses
    batch
        the data
    p
        the Lebesgue norm exponent

    return
    ------
    ndarray, shape (no. of elements in the batch, no. of hidden neurons)
        the product for each hidden neuron and each data sample.
    """
    coefficients = np.abs(weight_matrix) ** p
    return np.einsum("ik,jk,jk->ij", batch, weight_matrix, coefficients)


def plasticity_rule(weight_matrix, batch, R, p, first_indexes,
                    seventh_indexes, product, time_scale):
    #TODO
    

