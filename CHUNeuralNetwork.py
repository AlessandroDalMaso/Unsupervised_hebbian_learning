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


def product(weights, batch, p):
    """Multiply the inputs by the synapses weights of a single neuron.

    define coefficients, the multiply the weights, coefficients, and the data
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
    coefficients = np.abs(weights) ** p
    product = weights * coefficients * batch
    # k,k,ik->ik thanks to broadcasting
    return np.sum(product, axis=1)


def plasticity_rule_hebbian(product_result, single_weight, visible_neurons, R,
                            p, one_over_time_scale):
    subtrahend = R ** p * visible_neurons
    minuend = product_result * single_weight
    return (subtrahend - minuend) * one_over_time_scale


def plasticity_rule_anti(product_result, single_weight, visible_neurons, R, p,
                        delta, one_over_time_scale):
    subtrahend = R ** p * visible_neurons
    minuend = product_result * single_weight
    return (subtrahend - minuend) * (-delta) * one_over_time_scale
