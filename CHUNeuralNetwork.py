
from sklearn.base import TransformerMixin
import numpy as np
from math import sqrt
from scipy.integrate import solve_ivp

# %% defining external equaltions


def rank_finder(batch, weight_matrix, activation_function, k):
    """Return the indexes of the first and k-th most activated neurons."""
    hidden_neurons = hidden_neurons_func(batch, weight_matrix,
                                         activation_function)
    sorting = np.argsort(hidden_neurons)
    return (sorting[:, -1], sorting[:, -k])


def product(weight_vector, input_vector, p):
    """Multiply the inputs by the synapses weights of a single neuron.

    define coefficients, the multiply the weights, coefficients, and the data
    in a single operation.

    Parameters
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
    coefficients = np.abs(weight_vector) ** (p - 2)
    product = weight_vector * coefficients * input_vector
    return np.sum(product)


def plasticity_rule(weight_vector, input_vector, g, p, R, one_over_scale):
    """Calculate the update value for a row for weight_matrix.

    The update is zero for all but the most activated and the k-th most
    activated neuron.

    Parameters
    ----------
    weight_vector
        A row of weight_matrix to calculte the update for.
    input_vector
        the data sample
    first_index
        the index of the most activated neuron in weight_vector
    """
    product_result = product(weight_vector, input_vector, p)
    minuend = R ** p * input_vector
    subtrahend = product_result * weight_vector
    return g * (minuend - subtrahend) * one_over_scale

def plasticity_rule_vectorized(weight_matrix, batch, delta, p, R,
                               one_over_scale, indexes_hebbian, indexes_anti):
    """Calculate the update dW of weight_matrix.

    Each sample in batch updates only two rows of weight_matrix: the one
    corresponding to the most activated hidden neuron and the one corresponding
    to the k-th most activated.

    Parameters
    ----------
    weight_matrix
        The matrix to update.
    batch
        the data
    delta
        The relative strenght of anti-hebbian learning.
    p
        Lebesgue norm exponent.
    R
        The radius of the sphere at wich the hidden neurons will converge.
    one_over_scale
        One over the time scale of learning.
    indexes_hebbian
        The indexes of the hidden neurons wich will undergo hebbian learning.
    indexes_anti
        The indexes of the hidden neurons wich will undergo anti-hebbian
        learning.
    Return
    -----
    update
        ndarray, same shape as weight_matrix.
    """
    update = np.zeros(weight_matrix.shape)
    for i in range(len(batch)):

        j = indexes_hebbian[i]
        weight_vector_1 = weight_matrix[j]
        input_vector = batch[i]
        update[j] += plasticity_rule(weight_vector_1, input_vector, 1, p, R,
                                     one_over_scale)

        j2 = indexes_anti[i]
        weight_vector_2 = weight_matrix[j2]
        update[j2] += plasticity_rule(weight_vector_2, input_vector, -delta, p,
                                      R, one_over_scale)

    return update


def relu(currents):
    """Is the default activation function."""
    return np.where(currents < 0, 0, currents)


def hidden_neurons_func(batch, weight_matrix, activation_function):
    """Calculate hidden neurons activations."""
    currents = batch @ weight_matrix.T
    # ik,jk->ij
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


def ivp_helper(time, array, *args):
    """Is a version of plasticity_rule_vectorized compatible with solve_ivp."""
    (batch, delta, p, R, one_over_scale, indexes_hebbian, indexes_anti,
     dims) = args
    matrix = np.reshape(array, dims)
    update_matrix = plasticity_rule_vectorized(matrix, batch, delta, p, R,
                                               one_over_scale, indexes_hebbian,
                                               indexes_anti)
    return np.ravel(update_matrix)


# %% defining the class


class CHUNeuralNetwork(TransformerMixin):
    """Extract features from data using a biologically-inspired algorithm.

    Competing Hidden Units Neural Network. A 2-layers neural network
    that implements competition between patterns, learning unsupervised. The
    data transformed can then be used with a second, supervised, layer. See
    the article in the notes for a more complete explanation.

    Parameters
    ----------
        n_hiddens:
            the number of hidden neurons
        delta:
            Relative strenght of anti-hebbian learning.
        p:
            Exponent of the lebesgue norm used (see product function).
        R:
            Radius of the sphere on wich the weights will converge.
        n_of_input_neurons:
            the number of visible neurons (e.g. the number of features)
        one_over_scale:
            One over the time scale of learning.
        k:
            The k-th most activated hidden neuron will undergo anti-hebbian
            learning.
        activation_function:
            The activation function of the hidden neurons.

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

    def __init__(self, n_hiddens=2000, delta=0.4, p=3, R=1, scale=1, k=7,
                 activation_function=relu):
        self.n_hiddens = n_hiddens
        self.delta = delta
        self.p = p
        self.R = R
        self.one_over_scale = 1/scale
        self.k = k
        self.activation_function = relu

    def transform(self, X):
        """Transform the data."""
        return hidden_neurons_func(X, self.weight_matrix,
                                   self.activation_function)

    def fit(self, X, batch_size):
        """Fit the weigths to the data.

        Intialize the matrix of weights, the put the data in minibatches and
        update the matrix for each minibatch.

        Parameters
        ----------
        self
            The network itself.
        X
            The data to fit. Shape: (sample, feature).
        batch_size
            Number of elements in a batch.

        Return
        ------
        CHUNeuralNetwork
            The network itself.
        """
        dims = (self.n_hiddens, len(X[0]))
        self.weight_matrix = np.random.normal(0, 1/sqrt(self.n_hiddens), dims)
        # The weights are initialized with a gaussian distribution.
        x = 0
        for batch in batchize(X, batch_size):

            x += 1
            print(x)
            (indexes_hebbian, indexes_anti) = rank_finder(batch,
                                                          self.weight_matrix,
                                                          self.activation_function,
                                                          self.k)

            starting_array = np.ravel(self.weight_matrix)
            args = (batch, self.delta, self.p, self.R, self.one_over_scale,
                    indexes_hebbian, indexes_anti, dims)

            #bunch = solve_ivp(ivp_helper, (0, 1e4), starting_array,
            #                  method='RK45', args=args)

            #update_array = bunch.y[:,-1]
            #update_matrix = np.reshape(update_array, dims)
            update_matrix = plasticity_rule_vectorized(self.weight_matrix,
                                                       batch, self.delta,
                                                       self.p, self.R,
                                                       self.one_over_scale,
                                                       indexes_hebbian,
                                                       indexes_anti)
            self.weight_matrix += update_matrix
        return self

    def fit_transform(self, X, batch_size=2):
        """Fit the data, then transform it."""
        return self.fit(X, batch_size).transform(X)
