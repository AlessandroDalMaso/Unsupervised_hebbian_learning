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

    Notes:
    ------
        The name conventions for the variable is the same used in the article,
        when possible.

    References:
    ----------
        doi: 10.1073/pnas.1820458116
    """

# %% Defining main constants in the init function


    def __init__(self, n_of_input_neurons, n_of_hidden_neurons=200, p=3, k=7, delta=4, R=1,  # TODO k=7, K=2000
                 scale=1, batch_size=1):
        self.K = n_of_hidden_neurons
        self.J = n_of_input_neurons
        self.p = p
        self.k = k
        self.delta = delta
        self.R = R
        self.hidden_neurons = np.empty((batch_size, n_of_hidden_neurons))
        self.inputs = np.empty(n_of_input_neurons)
        self.weight_matrix = np.random.normal(0,
                                              1/math.sqrt(n_of_hidden_neurons),
                                              (n_of_hidden_neurons,
                                               n_of_input_neurons))
        # The weight initialization follows a convention i found online.
        self.scale = scale
        self.batch_size = batch_size

# %% Defining main equations and objects

    def product(self):
        shape = (self.batch_size, self.K, self.J, self.J)
        # an array of KxJ matrices containing 1-D arrays to be summed upon
        input_tensor = np.repeat(self.batch, self.K*self.J, axis=0)
        input_tensor = np.reshape(input_tensor, shape)
        weight_tensor = np.repeat(self.weight_matrix, self.batch_size * self.J,
                                  axis=0)
        weight_tensor = np.reshape(weight_tensor, shape)
        weights_abs = np.abs(weight_tensor)
        powers = input_tensor
        powers.fill(self.p-2)
        coefficients = np.power(weights_abs, powers)
        summatory = input_tensor * weight_tensor * coefficients
        result = np.sum(summatory, axis=-1)
        # summing along the last axis will give the correct shape
        assert result.shape == (self.batch_size, self.K, self.J)
        return result

    def g(self):
        """Return the update values for each neuron.

        Implements temporal competition between patterns.

        Return:
        ------
        ndarray
            the values that will be used in the plasticity_rule function
        """
        def ranking():
            columns = np.arange(0, self.batch_size, 1)
            sort = np.argsort(self.hidden_neurons)
            # sorts along last axis by default
            sort = sort.T
            # we want to identify want the biggest and k-th-est biggest value
            # from each row of the hidden_neurons matrix
            rows_biggest = sort[-1]
            rows_kth = sort[-self.k]
            result = np.zeros(self.hidden_neurons.shape)
            result[columns, rows_biggest] = 1
            result[columns, rows_kth] = -self.delta
            return result
        ranks = ranking()
        g_tensor = np.repeat(ranks, self.J, axis=-1)
        g_tensor = np.reshape(g_tensor, (self.batch_size, self.K, self.J))
        assert g_tensor[0][0][0] == g_tensor[0][0][1]
        return g_tensor

    def plasticity_rule(self):
        """Calculate dW for each element in the weight matrix.

        a sub-function returns dw for a single weight; the output is vectorized
        before being returned

        Returns
        -------
        ndarry:
            The increment of the weight, to be added to the weight itself.

        Notes
        -----
        Equation [3] of the article, with h as the argument of g().
        """
        inputs_tensor = np.repeat(self.batch, self.K, axis=0)
        inputs_tensor = np.reshape(inputs_tensor, (self.batch_size,
                                                   self.K, self.J))
        weights_tensor = np.repeat([self.weight_matrix], self.batch_size,
                                   axis=0)

        minuend = self.R ** self.p * inputs_tensor
        subtrahend = np.multiply(self.product(), weights_tensor)
        result = np.multiply(self.g(), (minuend - subtrahend))
        return np.sum(result, axis=0)  # summing over all results in the batch

    def radius(self):
        """Calculate a quantity that should converge to R.

        Elevate the absolute value of each hidden neuron (calculated for the
        first data point) to p, then sum these quantities toghether.

        Returns:
        --------
        Float
            the quantity descripted above.
        """
        powers = self.hidden_neurons[0]
        powers.fill(self.p)
        return np.sum(np.power(np.abs(self.hidden_neurons[0]), powers))



# %% implementing transform and fit methodss

    def transform(self, X):
        """Return the transformed array.

        Given raw input, this method will extract certain features, acting as
        the first layer of a neural network.

        Parameters
        ----------
        X:
            Array of the data in the format (n_samples, n_features)

        Return:
        ------
        ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        result = [self.weight_matrix.dot(x) for x in X]
        #result = np.empty((1, self.K))
        # ^ give the correct shape to result
        #for x in X:
        #    result = np.append(result, [self.weight_matrix.dot(x)], axis=0)
        #result = np.delete(result, 0, axis=0)  # delete the placeholder
        return result

    def fit(self, X, y=None):
        # TODO is it right to add y?
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

        def batch(iterable, n):
            lenght = len(iterable)
            for ndx in range(0, lenght, n):
                yield iterable[ndx:min(ndx + n, lenght)]

        for b in batch(X, self.batch_size):
            self.batch = b
            self.hidden_neurons = np.einsum("ij,kj->ki",
                                            self.weight_matrix, self.batch)
            # ^ dot product between each input vector and weight_matrix
            print(self.hidden_neurons[0][0])
            self.weight_matrix += self.plasticity_rule()
            # ^ updating the weight matrix
            a = input("batch processed, press enter to continue")

        return self
