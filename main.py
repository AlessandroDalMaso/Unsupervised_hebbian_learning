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

# %% Defining main constants

    """The convention for the constant names is the same of the article wich
    described the network implemented here. doi: 10.1073/pnas.1820458116"""

    def __init__(self, p=3, k=7, delta=4, n=4.5, R=1, w_ihn=1, K=3):
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
        hidden_neurons = np.empty(K)
        visible_neurons = np.empty(K)
        weight_matrix = np.random.normal(0, 1/math.sqrt(K), (K, K))
        # The weight initialization follows a convention i found online.
# %% Defining main equations and objects

    def product(self, X, Y, hidden_neuron_index):
        # TODO: is there a better way to pass indexes?
        """Define a product for later use.

        To each hidden neuron is associated a different product.

        Parameters
        ----------
        X, Y:
                the vectors to be multiplied
        hidden_neuron_index:
            the particular hidden neuron associated to the product.A different
            product is defined for each hidden neuron.

        Returns
        -------
        ndarray
            the product of X and Y as defined by the weight matrix
        """
        weights = self.weight_matrix[hidden_neuron_index].copy()
        for w in weights:
            w = abs(w) ** (self.p-2)
        summatory = X * weights * Y
        return (np.sum(summatory))

    def g(self, h):
        """Return a value used to update the weight matrix.

        Implements temporal competition between patterns.

        Parameters
        ----------
        h

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
        if rank == self.K - self.k:
            return -self.delta
        else:
            return 0

    def plasticity_rule(self,
                        hidden_neuron_index, visible_neuron_index, scale=1):
        """Calculate dW for a given weight.

        Given an hidden neuron calculates dW for a single weight W associated
        to a single visible neuron

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
        hni, vni = hidden_neuron_index, visible_neuron_index
        # aliasing for readability
        h = self.hidden_neurons[hni]  # hidden neuron value
        v = self.visible_neurons[vni]  # visible neuron value
        W = self.weight_matrix[hni, vni]
        # synapsis value
        weights = self.weight_matrix[hni].copy()
        # value of all the synapses of the hidden neuron

        factor = self.product(weights, self.visible_neurons, hni)

        return self.g(h) * (v * self.R ** self.p - factor * W) / scale

    def dynamical(self, hni, scale2=0.1):  # 8
        """Define the dynamics that lead to equilibrium.
        
        Implements a system of inhibition that will if iterated select one
        single pattern to emerge.
        
        Parameters
        ----------
        hni:
            hidden neuron index
        scale2:
            scale of the learning. should be smaller than the scale parameter
            of the plasticity_rule function.
        Returns:
        --------
        Float:
            The infinitesimal increment in the neuron value
        """
        h = self.hidden_neurons[hni]
        weights = self.weight_matrix[hni]
        minuend = self.product(weights, self.visible_neurons, hni)
        summatory = h - max(0, h)
        # ^ to respect the summation over all neurons but one

        for neuron in self.hidden_neurons:
            summatory += (max(neuron, 0) - neuron)

        return (minuend - self.w_inh * summatory)/scale2

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
        result = []
        for x in X:
            result.append(self.weight_matrix.dot(x))
        return result

    def fit(self, X, y=None): #TODO is it right to add y? complete docstring!
        """Fit the weights to the data provided.

        for each data point add to each weight the corresponding increment.

        Parameters:
        -----------
            X: the data to fit, in shape (n_samples, n_features)
        Returns:
        --------
        CHUNeuralNetwork:
            the network itself
        """
        for x in X:
            self.visible_neurons = x
            hidden_neurons = self.weight_matrix.dot(x)
            t = np.linspace(0, 1000, 1000)
            hn = integrate.odeint(np.vectorize(self.dynamical),
                                  hidden_neurons, t)
            hidden_neurons = hn[-1]
            for i in range(0, self.K-1):
                for j in range(0, self.K-1):
                    self.weight_matrix[i, j] += self.plasticity_rule(i, j)
        return self

CHUNeuralNetwork()

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