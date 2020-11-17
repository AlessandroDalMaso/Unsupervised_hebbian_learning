from sklearn.base import TransformerMixin
import numpy as np
from math import sqrt
from scipy.integrate import solve_ivp

# %% defining external equaltions


def rank_finder(batch, weight_matrix, activation_function, k):

    hidden_neurons = hidden_neurons_func(batch, weight_matrix,
                                         activation_function)

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
    coefficients = np.abs(weight_vector) ** (p -2)
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
    batch
        the data
    first_index
        the index of the most activated neuron in weight_vector
    
    """
    product_result = product(weight_vector, input_vector, p)
    minuend = R ** p * input_vector
    subtrahend = product_result * weight_vector
    return g * (minuend - subtrahend) * one_over_scale

def plasticity_rule_vectorized(weight_matrix, batch, delta, p, R,
                               one_over_scale, indexes_hebbian, indexes_anti):
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
    return np.where(currents < 0, 0, currents)


def hidden_neurons_func(batch, weight_matrix, activation_function):
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
    (batch, delta, p, R, one_over_scale, indexes_hebbian, indexes_anti,
     dims) = args
    matrix = np.reshape(array, dims)
    update_matrix = plasticity_rule_vectorized(matrix, batch, delta, p, R,
                                               one_over_scale, indexes_hebbian,
                                               indexes_anti)
    return np.ravel(update_matrix)
# %% defining the class


class CHUNeuralNetwork(TransformerMixin):
    """
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
        return hidden_neurons_func(X, self.weight_matrix,
                                   self.activation_function)

    def fit(self, X, batch_size=2):
        dims = (self.n_hiddens, len(X[0]))
        self.weight_matrix = np.random.normal(0, 1/sqrt(self.n_hiddens), dims)
        # The weights are initialized with a gaussian distribution.
        for batch in batchize(X, batch_size):

            (indexes_hebbian, indexes_anti) = rank_finder(batch,
                                                          self.weight_matrix,
                                                          self.activation_function,
                                                          self.k)

            starting_array = np.ravel(self.weight_matrix)
            args = (batch, self.delta, self.p, self.R, self.one_over_scale,
                    indexes_hebbian, indexes_anti, dims)

            bunch = solve_ivp(ivp_helper, (0, 1e4), starting_array,
                              method='RK45', args=args)

            update_array = bunch.y[:,-1]
            update_matrix = np.reshape(update_array, dims)
            self.weight_matrix += update_matrix
            return self

    def fit_transform(self, X, batch_size=2):
        return self.fit(X, batch_size).transform(X)