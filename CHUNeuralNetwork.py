"""
Created on Tue Jun 23 10:43:19 2020

Author: Alessandro Dal Maso
"""
import numpy as np
import math
from sklearn.base import TransformerMixin


class CHUNeuralNetwork(TransformerMixin):
    """Extract features from data using a biologically-inspired algorithm.

    Competing Hidden Units Neural Network. A 2-layers neural network
    that implements competition between patterns, learning unsupervised. The
    data transformed can then be used with a second, supervised, layer. See
    the article in the notes for a more complete explanation.

    Parameters
    ----------
        K:
            the number of hidden neurons
        J:
            the number of visible neurons (e.g. the number of features)
        p:
            exponent of the lebesgue norm used (se product function)
        k:
            The k-th unit will be weakened
        delta:
            regulates the weakening discussed above
        R:
            Radius of the sphere on wich the weights will converge
        w_inh:
            inhibition parameter
        hidden_neurons:
            initial values for those neurons.
        inputs:
            initial value for input neurons
        weight_matrix:
            the weights of the network.
        scale:
            a time scale.

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
                 delta=4, R=1, scale=1, batch_size=2):
        self.K = n_of_hidden_neurons
        self.J = n_of_input_neurons
        self.batch_size = batch_size
        self.p = p
        self.k = k
        self.delta = delta
        self.R = R
        self.hidden_neurons = np.zeros((batch_size, n_of_hidden_neurons))
        self.inputs = np.empty(n_of_input_neurons)
        self.weight_matrix = np.random.normal(0,
                                              1/math.sqrt(n_of_hidden_neurons),
                                              (n_of_hidden_neurons,
                                               n_of_input_neurons))
        # The weight initialization follows a convention i found online.
        self.scale = scale

# %% Defining main equations and objects

    def bigger1000(X):
        X = np.abs(X)
        X2 = np.where(X > 1000, 0, 1)
        return np.any(X2)
    def product(self):
        coefficients = np.abs(self.weight_matrix) ** (self.p - 2)
        subproduct = self.weight_matrix * coefficients
        return np.einsum("ij,kj->ki", subproduct, self.batch)
        # multiply every vector in the 2nd matrix by the 1st matrix.
        # the result shape is (batch_size, K)

    def g(self):
        sort = np.argsort(self.hidden_neurons)
        # sorts along last axis by default
        sort = sort.T
        # we want to identify the biggest and k-th-est biggest value
        # from each row of the hidden_neurons matrix
        rows_biggest = sort[-1]
        rows_kth = sort[-self.k]
        columns = np.arange(0, self.batch_size, 1)
        result = np.zeros(self.hidden_neurons.shape)
        result[columns, rows_biggest] = 1
        result[columns, rows_kth] = -self.delta
        return result
        # the result shape is (batch_size, K)

    def plasticity_rule(self):
        minuend = self.R ** self.p * self.batch
        minuend = np.repeat(minuend, self.K, axis=0)
        minuend = np.reshape(minuend, (self.batch_size, self.K, self.J))
        # i wish there was a way to do it whitout repeats
        subtrahend = np.einsum("ij,jk->ijk",
                               self.product(), self.weight_matrix)
        # multiply the n-th vector in the second matrix by the n-th scalar in
        # a vector of the the 1st matrix. repeat for each vector in the 2nd
        # matrix
        factor = minuend - subtrahend
        result = np.einsum("ij,ijk->jk", self.g(), factor)
        return result
        # multiply each weight of the synapsis w_ab relative to the hidden
        # neuron a and the input neuron b by g(a), wich only depends on the
        # value of the hidden neuron a. then sum over the batch to update the
        # weight matrix.

    def radius(self):  # TODO not useful in our implementation
        """Return a value that should converge to the R parameter.

        raise to the power of p the absolute values of the hidden neurons,
        then sum those values toghether.

        Returns
        -------
        float:
            the radius of convergence
        """
        return np.sum(np.abs(self.hidden_neurons[0]) ** self.p)

    def transform(self, X):
        return self.weight_matrix @ X.T

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

        def batchize(iterable, size):
            # credit: https://stackoverflow.com/users/3868326/kmaschta
            (x_i, y_i) = iterable.shape
            lenght = len(iterable)
            for n in range(0, lenght, size):
                yield iterable[n:min(n + size, lenght)]

        for b in batchize(X, self.batch_size):
            self.batch = b
            self.batch_size = np.size(b, 0)  # i hope this is right.
            self.hidden_neurons = np.einsum("ij,kj->ki",
                                            self.weight_matrix, self.batch)
            # ^ dot product between each input vector and weight_matrix
            self.weight_matrix += self.plasticity_rule()
            print(self.bigger1000(self.weight_matrix))
            # ^ updating the weight matrix
            # input("batch processed, press enter to continue")

        return self
