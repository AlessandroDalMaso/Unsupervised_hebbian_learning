"""
Created on Tue Jun 23 10:43:19 2020

Author: Alessandro Dal Maso
"""

import numpy as np
from scipy import integrate
from hypothesis import given
import hypothesis.strategies as st
import math
from sklearn.base import TransformerMixin


class CHUNeuralNetwork(TransformerMixin):
    """f."""  # TODO docstring

# %% Defining main constants in the init function

    """The convention for the constant names is the same of the article wich
    described the network implemented here. doi: 10.1073/pnas.1820458116"""

    def __init__(self, p=3, k=7, delta=4, n=4.5, R=1, w_ihn=1, K=3,
                 plasticity_scale=1, dynamical_scale=0.1):
        self.p = p  # Lebesgue norm exponent
        self.k = k
        # ^ The k-th unit and all successive units's synapses will be weakened
        self.delta = delta  # modulates the weakening
        self.n = n
        # ^ exponent used in the activation function in the supervised part
        self.R = R  # Radius of the sphere on wich the weights will converge
        self.w_inh = w_ihn  # TODO correct value?
        # TODO: these are all placeholders!
        self.K = K
        # n of hidden neurons, as well as visible neurons, following the
        # article's naming conventions
        self.hidden_neurons = np.empty(K)
        self.visible_neurons = np.empty(K)
        self.weight_matrix = np.random.normal(0, 1/math.sqrt(K), (K, K))
        # The weight initialization follows a convention i found online.
        self.plasticity_scale = plasticity_scale
        self.dynamical_scale = dynamical_scale

# %% Defining main equations and objects

    def product(self, weights, Y, X=None):
        # TODO: is there a better way to pass indexes?
        """Define a product for later use.

        To each hidden neuron is associated a different product.

        Parameters
        ----------
        X, Y:
                the vectors to be multiplied
        weights:
            the weights associated with an hidden neuron. A different product
            is defined for each hidden neuron.

        Returns
        -------
        ndarray
            the product of X and Y as defined by the weights.
        """
        # weights = self.weight_matrix[hidden_neuron_index].copy()
        weights_copy = weights.copy()
        for w in weights_copy:
            w = abs(w) ** (self.p-2)

        if X is None:
            X = weights.copy()

        totalsum = 0
        for (x, w, y) in zip(X, weights_copy, Y):
            totalsum += x*w*y
        return totalsum

    def g(self, h):
        """Return a value used to update the weight matrix.

        Implements temporal competition between patterns.

        Parameters
        ----------
        h:

        Returns
        -------
        float
            the value that will be used to update the weight matrix
        """
        rank = 0

        for neuron in self.hidden_neurons:
            if h >= neuron:
                rank += rank

        if rank == self.K:
            return 1
        if rank == self.K - self.k:  # TODO ask if correct procedure
            return -self.delta
        else:
            return 0

    def update_weights(self):
        """Calculate dW for each element in the weight matrix.

        a sub-function returns dw for a single weight; the output is vectorized
        before being returned

        Parameters
        ----------
        inputs:
            the input neurons value
        Returns
        -------
        ndarry:
            The increment of the weight, to be added to the weight itself.

        Notes
        -----
        Equation [3] of the article, with h as the argument of g().
        """
        def update_weights(weights):
            increments = np.empty(0)
            for (v, w, h) in zip(inputs, weights, self.hidden_neurons):
                factor = self.product(weights, self.visible_neurons)
                factor2 = v * self.R ** self.p - factor * w
                increment = self.g(h) * factor2 / self.plasticity_scale

                increments.append(increment)
            return increments

        return np.vectorize(update_weights)(self.weight_matrix)

    def neuron_dynamical_equation(self):  # 8
        """Define the dynamics that lead to equilibrium.

        Implements a system of inhibition that will if iterated select one
        single pattern to emerge.

        Parameters
        ----------
        inputs:
            visible neurons value
        Returns
        -------
        ndarray:
            numpy array of the increments dh
        Notes
        -----
        Equation [8] of the article.
        """
        def increment(weights, h):
            summatory = 0
            for ni, mu in self.hidden_neurons:
                if ni == mu:
                    pass
                else:
                    summatory += max(0, ni) - mu
            return self.product(weights, inputs) - self.w_inh * summatory
        return np.vectorize(increment)(self.weight_matrix, self.hidden_neurons)

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
        result = np.empty(0)
        for x in X:
            result.append(self.weight_matrix.dot(x))
        return result

    def fit(self, X, y=None):
        # TODO is it right to add y? complete docstring!
        """Fit the weights to the data provided.

        for each data point add to each weight the corresponding increment.

        Parameters
        ----------
            X: the data to fit, in shape (n_samples, n_features)

        Returns
        -------
        CHUNeuralNetwork:
            the network itself
        """
        for x in X:
            self.inputs = x
            self.hidden_neurons = self.weight_matrix.dot(x)
            time = np.linspace(0, 1000, 1000)  # for the differential equation
            states = integrate.odeint(self.neuron_dynamical_equation,
                                      hidden_neurons, time)
            self.hidden_neurons = states(-1)
            update_weights()
        return self


a = CHUNeuralNetwork()

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