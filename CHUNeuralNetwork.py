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
                            p, one_over_scale):
    """Calculate the update value for the most activated synapsis.

    Apply equation [3] of the original article only for the single most
    activated synapsis, but vectorized over all samples in the batch.

    Parameters
    ----------
    product_result
        The product between the synapse of the neuron and the input data, as
        defined by equation [2] of the original article.
    single_weight
        The weight of the synapsis.
    visible_neurons
        The i-th value for each sample in the data.
    R
        The radius of convergence.
    p
        The lebesgue norm exponent.
    one_over_scale:
        A parameter that modulates the evolution speed of the equation.

    Return
    ------
    ndarray, same shape as product_result and visible_neurons (1d)
        The update value for the most-activated synapsis.
    """
    subtrahend = R ** p * visible_neurons
    minuend = product_result * single_weight
    return (subtrahend - minuend) * one_over_scale


def plasticity_rule_anti(product_result, single_weight, visible_neurons, R, p,
                         one_over_scale, delta):
    """Calculate the update value for the k-th least activated synapsis.

    Apply equation [3] of the original article only for the k-th least
    activated synapsis, but vectorized over all samples in the batch.

    Parameters
    ----------
    product_result
        The product between the synapse of the neuron and the input data, as
        defined by equation [2] of the original article.
    single_weight
        The weight of the synapsis.
    visible_neurons
        The i-th value for each sample in the data
    R
        The radius of convergence.
    p
        The lebesgue norm exponent.
    one_over_scale:
        A parameter that modulates the evolution speed of the equation.
    delta:
        
    Return
    ------
    ndarray, same shape as product_result and visible_neurons (1d)
        The update value for the synapsis.
    """
    subtrahend = R ** p * visible_neurons
    minuend = product_result * single_weight
    return (subtrahend - minuend) * (-delta) * one_over_scale
