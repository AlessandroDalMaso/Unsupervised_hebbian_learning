"""
Created on Tue Jun 23 10:43:19 2020.

Author: Alessandro Dal Maso
"""
import numpy as np
from pickle import dump
from math import sqrt
from sklearn.base import TransformerMixin
from scipy.integrate import odeint, solve_ivp


# %% defining external functions


def product(weight_matrix, batch, p, save_matrices):
    """Return a matrix that contains the product from the original article.

    This matrix contains the products of the input neurons with the synapses
    of each hidden neurons. it is not the euclidean product between these two
    quantities but the product defined in equation [2] of the original article.

    Parameters
    ----------
    weight_matrix
        the matrix of the synapses of the hidden neurons.
    batch
        the value of the input neurons.
    p
        the Lebesgue exponent used to define the product.
    save_matrices
        a debugging boolean option.

    Returns
    -------
    ndarray
        the matrix of all scalar products between input neurons and the
        synapses of a single hidden neuron.
    """
    coefficients = np.abs(weight_matrix) ** (p - 2)
    result = np.einsum("jk,jk,ik->ij", weight_matrix, coefficients, batch)
    if save_matrices:
        # for testing purposes. won't be executed in the final version
        product_r = open('./product_r', 'wb')
        dump(result, product_r)
        product_r.close()
    return result


def g(hidden_neurons, k, delta, save_matrices):
    """Return a coefficient that modulates hebbian and anti-hebbian learning.

    Return a matrix with the same shape of hidden_neurons with zero,
    positive and negative values.

    Parameters
    ----------
    hidden_neurons
        the matrix (one vector per element in the batch) of hidden neurons
    k
        the k-th-est biggest value neuron will be suppressed...
    delta
        ...by this amount.
    save_matrices
        a debugging boolean option.

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
    columns = np.arange(0, len(hidden_neurons), 1)
    result = np.zeros(hidden_neurons.shape)
    result[columns, rows_biggest] = 1
    result[columns, rows_kth] = -delta
    if save_matrices:
        # for testing purposes. won't be executed in the final version
        g_r = open('./g_r', 'wb')
        dump(result, g_r)
        g_r.close()
    return result
    # the result shape is (batch_size, K)


def plasticity_rule(weight_matrix, R, p, batch, scale, save_matrices,
                    hidden_neurons, k, delta):
    """Return the value used to update the weight matrix.

    Corresponds to equation [3] of the article, but substituting the hidden
    neuron value to Q.

    Parameters
    ----------
    weight_matrix
        the matrix of the synapses of the hidden neurons.
    R
        The radius at wich the product between the synapses of each hidden
        neuron and the input neurons will converge.
    p
        the Lebesgue exponent used to define the product.
    batch
        the value of the input neurons.
    scale
        a scaling parameter.
    save_matrices
        a debugging boolean option.
    hidden_neurons:
        the matrix (one vector per element in the batch) of hidden neurons
    k
        the k-th-est biggest value neuron will be suppressed...
    delta
        ...by this amount.

    Returns
    -------
    ndarray
        the value to be added to the weight matrix.
    """
    g_result = g(hidden_neurons, k, delta, save_matrices)
    product_result = product(weight_matrix, batch, p, save_matrices)
    minuend = R ** p * np.einsum("ij,ik->jk", g_result, batch)
    subtrahend = np.einsum("ij,ij,jk->jk", g_result, product_result,
                           weight_matrix)

    result = (minuend - subtrahend) / scale

    if save_matrices:
        # for testing purposes. won't be executed in the final version
        plasticity_r = open('./plasticity_r', 'wb')
        dump(result, plasticity_r)
        plasticity_r.close()
    return result
    # multiply each weight of the synapsis w_ab relative to the hidden
    # neuron a and the input neuron b by g(a), wich only depends on the
    # value of the hidden neuron a. then sum over the batch to update the
    # weight


def linear_plasticity_rule(time, weight_array, R, p, batch, scale,
                           save_matrices, hidden_neurons, k, delta,
                           n_of_hidden_neurons, n_of_input_neurons):
    """Return the value used to update the weight matrix.

    Does the same as plasticity_rule, but takes a time vector and a weight
    array instead of a matrix to be compatible with numpy.integrate.odeint.
    Corresponds to equation [3] of the article, but substituting the hidden
    neuron value to Q.

    Parameters
    ----------
    time
        the integration time of the differential equation
    weight_array
        the matrix of the synapses of the hidden neurons, reshaped to an array.
    R
        The radius at wich the product between the synapses of each hidden
        neuron and the input neurons will converge.
    p
        the Lebesgue exponent used to define the product.
    batch
        the value of the input neurons.
    scale
        a scaling parameter.
    save_matrices
        a debugging boolean option.
    hidden_neurons:
        the matrix (one vector per element in the batch) of hidden neurons
    k
        the k-th-est biggest value neuron will be suppressed...
    delta
        ...by this amount.
    n_of_hidden_neurons:
        the number of hidden neurons in the network.
    n_of_input_neurons:
        the number of input neurons in the network.

    Returns
    -------
    ndarray
        the value to be added to the weight matrix.
    """
    shape = (n_of_hidden_neurons, n_of_input_neurons)
    print(weight_array)
    weight_matrix = np.reshape(weight_array, shape)

    update_matrix = plasticity_rule(weight_matrix, R, p, batch, scale,
                                    save_matrices, hidden_neurons, k, delta)

    update_array = np.ravel(update_matrix)
    return update_array


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
    ------
    """
    # credit: https://stackoverflow.com/users/3868326/kmaschta
    (x_i, y_i) = iterable.shape
    lenght = len(iterable)
    for n in range(0, lenght, size):
        yield iterable[n:min(n + size, lenght)]


def ReLU(matrix):
    """Return the maximum between zero and the argument value for each scalar.

    Parameters
    ----------
    matrix
        The input currents of the neural network
    Return
    ------
    ndarray
        the hidden neurons value
    """
    return np.maximum(matrix, 0)


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
        As this network is composed of only two layers, the hidden neurons
        aren't actually hidden, but will be in practice as this network is
        meant to be used in conjunction with a second, supervised network that
        will take the hidden neurons layer as its input layer.

    References
    ----------
        doi: 10.1073/pnas.1820458116
    """

# %% Defining main constants in the init function

    def __init__(self, n_of_hidden_neurons=5, p=3, k=2,
                 delta=0.4, R=1, scale=1, activation_function = ReLU):
        self.p = p
        self.k = k
        self.delta = delta
        self.R = R
        self.scale = scale
        self.n_of_hidden_neurons = n_of_hidden_neurons
        self.activation_function = activation_function


# %% Defining main equations and objects

    def activation_function(hidden_neurons):
        """Return the hidden neuron value if positive, zero otherwise.

        Rectifier Linear Unit activation function.

        Parameters
        ----------
        hidden_neurons
            the neuron layer to activate
        Return
        ------
        ndarray
            The activation values.
        """
        return np.where(hidden_neurons < 0, 0, hidden_neurons)

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
        # TODO: check the behavior of other scikit neural networks. how do they
        # behave when no fitting is done?

    def fit(self, X, batch_size=2, save_matrices=False, y=None):
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
        (n_of_samples, n_of_input_neurons) = X.shape

        self.weight_matrix = np.random.normal(0,
                                              1/sqrt(self.n_of_hidden_neurons),
                                              (self.n_of_hidden_neurons,
                                               n_of_input_neurons))
        # The weights are initialized with a gaussian distribution.

        for batch in batchize(X, batch_size):
            currents = np.einsum("jk,ik->ij", self.weight_matrix, batch)
            # ^ dot product between each input vector and weight_matrix
            hidden_neurons = self.activation_function(currents)
            weight_array = np.ravel(self.weight_matrix)

            args = (self.R, self.p, batch, self.scale, save_matrices,
                    hidden_neurons, self.k, self.delta,
                    self.n_of_hidden_neurons, n_of_input_neurons)

            # odeint_result = odeint(linear_plasticity_rule, weight_array,
            #                       [0, 1e5], args, tfirst=True)

            # update = np.reshape(odeint_result[1], self.weight_matrix.shape)
            bunch = solve_ivp(linear_plasticity_rule, (0, 1e5), weight_array,
                              method='RK45', args=args)
            result = bunch.y[:, -1]
            update = np.reshape(result, self.weight_matrix.shape)
            self.weight_matrix += update
            # updating the weight matrix

            if save_matrices:
                # for testing purposes. won't be executed in the final version
                weights_r = open('./weights_r', 'wb')
                dump(self.weight_matrix, weights_r)
                weights_r.close()
                hidden_neurons_r = open('./hidden_r', 'wb')
                dump(hidden_neurons, hidden_neurons_r)
                hidden_neurons_r.close()

        return self
