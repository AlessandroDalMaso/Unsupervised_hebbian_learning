from sklearn.base import TransformerMixin
import numpy as np
from math import sqrt
from scipy.integrate import odeint


# %% defining external equaltions


def rank_finder(hidden_neurons, k):
    sorting = np.argsort(hidden_neurons)
    return (sorting[:,-1], sorting[:,-k])


def product(weight_vector, input_vector, p):
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
    coefficients = np.abs(weight_vector) ** p
    product = weight_vector * coefficients * input_vector
    return np.sum(product, axis=1)


def plasticity_rule(weight_vector, input_vector, g, p, R, one_over_scale):
    """Calculate the update value for a row for weight_matrix.

    The update is zero for all but the most activated and the k-th most
    activated neuron.

    Parameters
    ----------
    weight_vector
        A row of weight_matrix to calculte the update for.
    batch
        the data
    first_index
        the index of the most activated neuron in weight_vector
    
    """
    product_result = product(weight_vector, input_vector, p)
    minuend = R ** p * input_vector
    subtrahend = product_result * weight_vector
    return g * (minuend - subtrahend) * one_over_scale

def plasticity_rule_vectorized(weight_matrix, batch, g, p, R, one_over_scale,
                               indexes_hebbian, indexes_anti):
    update = np.zeros(weight_matrix.shape)
    for i in range(len(batch)):

        j = indexes_hebbian[i]
        weight_vector = weight_matrix[j]
        input_vector = batch[i]
        update[j] += plasticity_rule(weight_vector, input_vector, 1, p, R,
                                    one_over_scale)

        j2 = indexes_anti[i]
        weight_vector = weight_matrix[j2]
        update[j2] += plasticity_rule(weight_vector, input_vector, 1, p, R,
                                    one_over_scale)

    return update


def relu(currents):
    return np.where(currents < 0, 0, currents)


def hidden_neurons_func(batch, weight_matrix, activation_function):
    currents = np.einsum("ik,jk->ij", batch, weight_matrix)
    return activation_function(currents)


def batchize(iterable, size):
    """Put iterables in batches.

    Returns a new iterable wich yelds an array of the argument iterable in a
    list.

    Parameters
    ----------
    iterable:
        the iterable to be batchized.
    size:
        the number of elements in a batch.

    Return
    ------
    iterable
        of wich each element is an n-sized list of the argument iterable.

    Notes
    -----
    credit: https://stackoverflow.com/users/3868326/kmaschta
    """
    lenght = len(iterable)
    for n in range(0, lenght, size):
        yield iterable[n:min(n + size, lenght)]
