"""
Created on Tue Jun 23 10:43:19 2020

Author: Alessandro Dal Maso
"""
x = 0
import numpy as np
from scipy import integrate
from hypothesis import given
import hypothesis.strategies as st
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


    def __init__(self, J, K=2000, p=3, k=7, delta=4, R=1, w_ihn=1,
                 scale=1, batch_size=1):
        self.K = K
        self.J = J
        self.p = p
        self.k = k
        self.delta = delta
        self.R = R
        self.w_inh = w_ihn
        self.hidden_neurons = np.empty(K)
        self.inputs = np.empty(J)
        self.weight_matrix = np.random.normal(0, 1/math.sqrt(K), (K, J))
        # The weight initialization follows a convention i found online.
        self.scale = scale
        self.batch_size = 1

# %% Defining main equations and objects

    def product(self):
        shape = (self.batch_size, self.K, self.J, self.J)
        input_tensor = np.tile(self.batch, (1, self.K*self.J))
        input_tensor = np.reshape(input_tensor, shape)
        weight_tensor = np.tile(self.weight_matrix,
                                (1, self.batch_size * self.J))
        weight_tensor = np.reshape(weight_tensor, shape)
        weights_abs = np.abs(weight_tensor)
        powers = input_tensor.fill(self.p-2)
        coefficients = np.power(weights_abs, powers)
        summatory = input_tensor * weight_tensor * coefficients
        result = np.sum(summatory, axis=-1)
        assert result.shape == (self.batch_size, self.K, self.J)
        return result

    def g(self):
        """Return the update values for each neuron.

        Implements temporal competition between patterns.

        Return:
        ------
        ndarray
            the values that will be used in the plasicity_rule function
        """
        def ranking():
            result = np.zeros(self.hidden_neurons.shape)
            sortvalues = np.argsort(self.hidden_neurons)
            # from lowest to highest
            result[sortvalues[self.K-1]] = 1
            result[sortvalues[self.k]] = -self.delta
            return result
        ranks = ranking()
        g_tensor = np.repeat(ranks, self.J, axis=-1)
        g_tensor = np.reshape(g_tensor, (self.K, self.J))
        assert g_tensor[0][0] == g_tensor[0][1]
        g_tensor = np.repeat([g_tensor], self.batch_size, axis=0)
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
        inputs_tensor = np.reshape(inputs_tensor,
                                   (self.batch_size, self.K, self.J))
        weights_tensor = np.repeat([self.weight_matrix],
                                   self.batch_size, axis=0)

        minuend = self.R ** self.p * inputs_tensor
        subtrahend = np.multiply(self.product(), weights_tensor)
        result = self.g() * (minuend - subtrahend)
        return np.sum(result, axis=-1)  # summing over all results in the batch



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
        result = np.empty((1, self.K))
        # ^ give the correct shape to result
        for x in X:
            result = np.append(result, [self.weight_matrix.dot(x)], axis=0)
        result = np.delete(result, 0, axis=0)  # delete the placeholder
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
        global x
        for b in batch(X, self.batch_size):
            x += 1
            print(x)
            #print("questo è il valore di un neurone", self.hidden_neurons[1])
            self.batch = b
            #print("questo è il valore di un input", self.inputs[1])
            self.weight_matrix += self.plasticity_rule()
        return self
