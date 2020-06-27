# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 10:43:19 2020

@author: Alessandro Dal Maso
"""
import numpy as np
import math
# %% Defining some constants

lebesgue_norm_exp = 3
k = 7
delta = 4
n = 4.5
sphere_radius = 1
# TODO : better names

# %% Defining main equations and objects


def product(X, Y, neuron_index):
    # TODO: is there a better way to pass indexes?
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
    weights_copy = weights[neuron_index]
    for w in weights_copy:
        w = abs(w) ** (lebesgue_norm_exp-2)
    summatory = X * w * Y
    return (np.sum(summatory))


def learning_activation_function(neuron):  # TODO: is it the correct name? g
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


def weight_infinitesimal_change(time_scale,
                                hidden_neuron_index, inputs, weight):
    """Calculate dW for a given weight.

    Given an hidden neuron calculates dW for a single weight W associated to a
    single visible neuron

    Parameters
    ----------
    time-scale:
        the time scale of the learning dynamic
    hidden_neuron_index:
        the index of the hidden neuron
    inputs:
        the value of visible neurons
    weight:
        the value of the weight to be updated.

    Returns
    -------
    Float:
        The increment of the weight, to be added to the weight itself.
    """
    Q = product(weights[hidden_neuron_index], inputs, hidden_neuron_index)\
        / product(weights[hidden_neuron_index], weights[hidden_neuron_index],
                  hidden_neuron_index)\
        ** ((lebesgue_norm_exp-1) / lebesgue_norm_exp)

    return learning_activation_function(Q)\
        * (inputs[hidden_neuron_index] * sphere_radius ** lebesgue_norm_exp
           - product(weights[hidden_neuron_index], inputs,
                     hidden_neuron_index) * weight) / time_scale


# %% Implementation


# TODO: Initialize the weight_matrix with random gaussian weights,
# with 1/sqrt(inputs) sigma, a convention i found online
n_of_input_neurons = 3  # TODO: these are all placeholders!
n_of_hidden_neurons = 3
weights = np.ones((n_of_input_neurons, n_of_hidden_neurons))

# %% testing


