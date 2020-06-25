# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 10:43:19 2020

@author: Alessandro Dal Maso
"""
import numpy as np

# %% Defining some constants

lebesgue_norm_exp = 3
k = 7
delta = 4
n = 4.5
sphere_radius = 1
# TODO : better names

# %% Defining main equations and objects


def product(X, Y, neuron_index):
    """Define a product for later use.

    To each hidden neuron is associated a different product.

    Parameters
    ----------
    X, Y:
            the vectors to be multiplied
    neuron_index:
        the particular hidden neuron associated to the product.

    Returns
    -------
    ndarray
        the product of X and Y as defined by the weight matrix
    """
    product_matrix = np.eye * np.linalign.norm(weights[neuron_index])\
        ** (lebesgue_norm_exp-2)
    return (X * product_matrix * Y)


def learning_activation_function(neuron):  # TODO: is it the correct name?
    """Return a value used to update the weight matrix.

    Implements temporal competition between patterns.

    Parameters
    ----------
    neuron
        the activation function of the hidden neuron

    Returns
    -------
    float
        the value that will be used to update the weight matrix
    """
    if neuron < 0:
        return 0
    if 7 < neuron:  # TODO placeholder value
        return 1
    return -1*delta


def weight_infinitesimal_change(time_scale, input_neurons, weights,
                                weight_index):
    """f."""
    Q = product(input_neurons, weights) / product(weights, weights) ** \
        (lebesgue_norm_exp - 1/lebesgue_norm_exp)
    return 0  # TODO finish function


# %% Implementation

# TODO: Initialize the weight_matrix with random gaussian weights,
# with 1/sqrt(inputs) sigma, a convention i found online
n_of_input_neurons = 3  # TODO: these are all placeholders!
n_of_hidden_neurons = 3
weights = np.ones(n_of_input_neurons, n_of_hidden_neurons)
