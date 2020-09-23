"""
Created on Tue Jun 23 10:43:19 2020

Author: Alessandro Dal Maso
"""
import numpy as np
import math
from sklearn.base import TransformerMixin
import pickle as pk

# %% defining external functions

def bigger10000(matrix):
    X = np.abs(matrix)
    bool_matrix = X > 10000
    return np.any(bool_matrix)


def product(weight_matrix, p, save_matrices, batch):
    """Define a product for later use.

    TODO
    """
    coefficients = np.abs(weight_matrix) ** (p - 2)
    subproduct = weight_matrix * coefficients
    result = np.einsum("ij,kj->ki", subproduct, batch)
    # multiply every vector in the 2nd matrix by the 1st matrix.
    # the result shape is (batch_size, K)
    if save_matrices:
        product_r = open('./product_r', 'wb')
        pk.dump(result, product_r)
        product_r.close()
    return result


def g(hidden_neurons, k, batch_size, delta, save_matrices):
    """Return a coefficient that modulates hebbian and anti-hebbian learning.

    Return a matrix with the same shape of hidden_neurons with zero,
    positive and negative values.

    Returns
    -------
    ndarray
        the coefficient that modulates hebbian learning.

    Notes
    -----
    the implementation is the same described in the "a fast implementation"
    section of the reference article.
    """
    sort = np.argsort(hidden_neurons)
    # sorts along last axis by default
    sort = sort.T
    # we want to identify the biggest and k-th-est biggest value
    # from each row of the hidden_neurons matrix
    rows_biggest = sort[-1]
    rows_kth = sort[-k]
    columns = np.arange(0, batch_size, 1)
    result = np.zeros(hidden_neurons.shape)
    result[columns, rows_biggest] = 1
    result[columns, rows_kth] = -delta
    if save_matrices:
        g_r = open('./g_r', 'wb')
        pk.dump(result, g_r)
        g_r.close()
    return result
    # the result shape is (batch_size, K)


def plasticity_rule(batch_size, n_of_hidden_neurons, n_of_input_neurons, batch,
                    R, p, weight_matrix, scale, save_matrices, hidden_neurons,
                    k, delta):
    """Returns the value used to update the weight matrix

    Corresponds to equation [3] of the article, but substituting the hidden
    neuron value to Q.

    Returns
    -------
    ndarray
        the value to be added to the weight matrix.
    """
    shape = (batch_size, n_of_hidden_neurons,
             n_of_input_neurons)  # for later use

    minuend = R ** p * batch
    minuend = np.repeat(minuend, n_of_hidden_neurons, axis=0)
    minuend = np.reshape(minuend, shape)
    # i wish there was a way to do it whitout repeats
    subtrahend = np.einsum("ij,jk->ijk", product(weight_matrix, p,
                                                 save_matrices, batch),
                           weight_matrix)
    # multiply the n-th vector in the second matrix by the n-th scalar in
    # a vector of the the 1st matrix. repeat for each vector in the 2nd
    # matrix
    factor = minuend - subtrahend
    result = np.einsum("ij,ijk->jk", g(hidden_neurons, k, batch_size, delta,
                                       save_matrices), factor) / scale
    if save_matrices:
        plasticity_r = open('./plasticity_r', 'wb')
        pk.dump(result, plasticity_r)
        plasticity_r.close()
    return result
    # multiply each weight of the synapsis w_ab relative to the hidden
    # neuron a and the input neuron b by g(a), wich only depends on the
    # value of the hidden neuron a. then sum over the batch to update the
    # weight


def batchize(iterable, size):
    """Put iterables in batches.

    Returns a new iterable wich yelds an array of the argument iterable in a
    list.

    Parameters
    ----------
    iterable:
        the iterable to be batchized.
    size:
        the number of elements in a barch.

    Return
    TODO ask professor
    ------
    """
    # credit: https://stackoverflow.com/users/3868326/kmaschta
    (x_i, y_i) = iterable.shape
    lenght = len(iterable)
    for n in range(0, lenght, size):
        yield iterable[n:min(n + size, lenght)]


# %% defining the class

class CHUNeuralNetwork(TransformerMixin):
    """Extract features from data using a biologically-inspired algorithm.

    Competing Hidden Units Neural Network. A 2-layers neural network
    that implements competition between patterns, learning unsupervised. The
    data transformed can then be used with a second, supervised, layer. See
    the article in the notes for a more complete explanation.

    Parameters
    ----------
        n_of_hidden_neurons:
            the number of hidden neurons
        n_of_input_neurons:
            the number of visible neurons (e.g. the number of features)
        p:
            exponent of the lebesgue norm used (se product function)
        k:
            The k-th unit will be weakened
        delta:
            regulates the weakening discussed above
        R:
            Radius of the sphere on wich the weights will converge
        scale:
            a time scale.
        batch_size:
            the size of the minibatches in wich the data will be processed.

    Notes
    -----
        The name conventions for the variable is the same used in the article,
        when possible.

    References
    ----------
        doi: 10.1073/pnas.1820458116
    """

# %% Defining main constants in the init function

    def __init__(self, n_of_input_neurons, n_of_hidden_neurons=2000, p=3, k=7,
                 delta=0.4, R=1, scale=1, batch_size=2, save_matrices=False):
        self.n_of_hidden_neurons = n_of_hidden_neurons
        self.n_of_input_neurons = n_of_input_neurons
        self.batch_size = batch_size
        self.p = p
        self.k = k
        self.delta = delta
        self.R = R
        self.hidden_neurons = np.zeros((batch_size, n_of_hidden_neurons))
        # each 1-D array is calculated for a different element of the input
        # batch
        self.inputs = np.empty(n_of_input_neurons)
        self.weight_matrix = np.random.normal(0,
                                              1/math.sqrt(n_of_hidden_neurons),
                                              (n_of_hidden_neurons,
                                               n_of_input_neurons))
        # The weights are initialized with a gaussian distribution.
        self.scale = scale
        self.save_matrices = save_matrices

# %% Defining main equations and objects

    def transform(self, data):
        """Process the raw data by multiplying them by the weight matrix.

        Return the data preprocessed, to be used as input for a supervised
        learning layer.

        Parameters
        ----------
        data
            the raw data matrix

        Returns
        -------
        ndarray
            the extracted features data
        """
        return self.weight_matrix @ data.T

    def fit(self, X, y=None):
        """Fit the weights to the data provided.

        for each data point add to each weight the corresponding increment.

        Parameters
        ----------
        X: the data to fit, in shape (n_samples, n_features)
        y: as this is unsupervised learning should always be None
        Returns
        -------
        CHUNeuralNetwork:
            the network itself
        """
        for b in batchize(X, self.batch_size):
            self.batch = b
            self.batch_size = np.size(b, 0)
            if self.save_matrices:
                weights_r = open('./weights_r', 'wb')
                pk.dump(self.weight_matrix, weights_r)
                weights_r.close()
                hidden_neurons_r = open('./hidden_r', 'wb')
                pk.dump(self.hidden_neurons, hidden_neurons_r)
                hidden_neurons_r.close()
            self.hidden_neurons = np.einsum("ij,kj->ki", self.weight_matrix,
                                            self.batch)
            # ^ dot product between each input vector and weight_matrix
            self.weight_matrix += plasticity_rule(self.batch_size,
                                                  self.n_of_hidden_neurons,
                                                  self.n_of_input_neurons,
                                                  self.batch,
                                                  self.R, self.p,
                                                  self.weight_matrix,
                                                  self.scale,
                                                  self.save_matrices,
                                                  self.hidden_neurons,
                                                  self.k, self.delta)
            # ^ updating the weight matrix
        return self
