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


def plasticity_rule(time, weights, batch, first_index, seventh_index,
                    time_scale, R, p, delta):
    product_result = product(weights, batch, p)
    v_1 = batch[:,first_index]
    w_1 = weights[first_index]
    v_7 = batch[:,seventh_index]
    w_7 = weights[seventh_index]
    result_1 = R ** p * v_1 - product_result * w_1
    result_7 = -1 * delta * (R ** p * v_7 - product_result * w_7)
    update = np.zeros(weights.size)
    update[first_index] = result_1
    update[seventh_index] = result_7
