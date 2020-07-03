# -*- coding: utf-8 -*-.
"""
Created on Tue Jun 23 10:43:19 2020

Author: Alessandro Dal Maso

"""

import numpy as np
from hypothesis import given
import hypothesis.strategies as st

# %% Defining main constants

"""The convention for the constant names is the same of the article wich
described the network implemented here. doi: 10.1073/pnas.1820458116"""

p = 3  # Lebesgue norm exponent
k = 7  # The k-th unit and all successive units's synapses will be weakened
delta = 4  # modulates the weakening
n = 4.5  # exponent used in the activation function in the supervised part
R = 1  # Radius of the sphere on wich the weights will converge

# %% Defining main equations and objects


def product(X, Y, hidden_neuron_index):
    # TODO: is there a better way to pass indexes?
    """Define a product for later use.

    To each hidden neuron is associated a different product.

    Parameters
    ----------
    X, Y:
            the vectors to be multiplied
    hidden_neuron_index:
        the particular hidden neuron associated to the product.A different
        prouduct is defined for each hidden neuron.

    Returns
    -------
    ndarray
        the product of X and Y as defined by the weight matrix
    """
    weights = weight_matrix[hidden_neuron_index].copy()
    for w in weights:
        w = abs(w) ** (p-2)
    summatory = X * w * Y
    return (np.sum(summatory))


def g(neuron):  # Following the name convention of the article
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


def plasticity_rule(time_scale, hidden_neuron_index, visible_neuron_index):
    """Calculate dW for a given weight.

    Given an hidden neuron calculates dW for a single weight W associated to a
    single visible neuron

    Parameters
    ----------
    time-scale:
        the time scale of the learning dynamic
    hidden_neuron_index:
        the index of the hidden neuron
    visible_neuron_index:
        the index of the visible neuron

    Returns
    -------
    Float:
        The increment of the weight, to be added to the weight itself.

    Notes
    -----
    Equation [3] of the article, with h as the argument of g().
    """
    h = hidden_neurons[hidden_neuron_index]
    v = visible_neurons[visible_neuron_index]
    W = weight_matrix[visible_neuron_index, hidden_neuron_index]
    weights = weight_matrix[hidden_neuron_index].copy()

    factor = product(weights, visible_neurons, hidden_neuron_index)

    return g(h) * (v * R ** p - factor * W) / time_scale


# %% Implementation


# TODO: Initialize the weight_matrix with random gaussian weights,
# with 1/sqrt(inputs) sigma, a convention i found online
n_of_visible_neurons = 3  # TODO: these are all placeholders!
n_of_hidden_neurons = 3
weight_matrix = np.ones((n_of_visible_neurons, n_of_hidden_neurons))
hidden_neurons = np.ones(n_of_hidden_neurons)
visible_neurons = np.ones(n_of_visible_neurons)
# %% testing

product([1.1 , 1, 2],[1.1,1,2],2)
"""
def test_product_proportionality(x = [1.1,1,2], y = [1.3,1.1,1], z = 2):
    product(x,y,z)

@given(array1=hn.arrays(float, n_of_input_neurons),
       array2=hn.arrays(float, n_of_input_neurons),
       # TODO i don't understand why but i can't define both arrays with a
       # comma!
       index=st.integers(min_value=0, max_value=n_of_input_neurons-1),
       proportionality_constant=st.floats())
def test_product_proportionality(array1, array2, index,
                                 proportionality_constant):
    h.
    assert product(array1 * proportionality_constant, array2, index)\
        == product(array1, array2, index) * proportionality_constant

    assert product(array1, array2 * proportionality_constant, index)\
        == product(array1, array2, index) * proportionality_constant

"""