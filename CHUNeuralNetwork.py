"""A biology-inspired data transformer."""

from sklearn.base import TransformerMixin
import numpy as np
from math import sqrt
from scipy.integrate import solve_ivp

# %% defining external equaltions


def norms(matrix, p):
    """p-norms of vectors in a matrix."""
    return np.sum(np.abs(matrix) ** p, axis=1)


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


def scale_update(update, epoch, epochs, learn_rate=0.2):
    max_norm = np.amax(np.abs(update))
    esp = learn_rate*(1-epoch/epochs)
    return esp*update/max_norm


def relu(currents):
    """Is the default activation function."""
    return np.where(currents < 0, 0, currents)

def hidden_neurons_func(batch, weight_matrix, activation_function):
    """Calculate hidden neurons activations."""
    currents = batch @ np.transpose(weight_matrix)
    #currents2 = np.einsum("ik,jk->ij", batch, weight_matrix)
    #currents3 = np.einsum("ik,kj->ij", batch, weight_matrix.T)
    # TODO explain the different behavior
    return activation_function(currents)


def hidden_neurons_func_2(batch, weight_matrix, p):
    product = weight_matrix * np.abs(weight_matrix) ** p-2
    return product @ (batch.T)


def ranker(batch, weight_matrix, activation_function, k, p):
    """Return the indexes of the first and k-th most activated neurons."""
    hidden_neurons = hidden_neurons_func_2(batch, weight_matrix,
                                         p)
    sorting = np.argsort(hidden_neurons)
    return (sorting[:, -1], sorting[:, -k])


def product(weight_vector, input_vector, p):
    """Multiply the inputs by the synapses weights of a single neuron.

    define coefficients, then multiply the weights, coefficients, and the data
    in a single operation.

    Parameters
    ----------
    weight_vector
        The vector of the synapses weights.
    input_vector
        The data sample.
    p
        The Lebesgue norm exponent.

    Return
    ------
    ndarray, shape (no. of elements in the batch, no. of hidden neurons)
        the product for each hidden neuron and each data sample.

    Notes
    -----
    Equation [2] of the referenced article.
    """
    coefficients = np.abs(weight_vector) ** (p - 2)
    product = weight_vector * coefficients * input_vector
    return np.sum(product)


def plasticity_rule(weight_vector, input_vector, g, p, R, one_over_scale):
    """Calculate the update value for a single row for weight_matrix.

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
    row_update = g * (minuend - subtrahend) * one_over_scale
    return row_update


def plasticity_rule_vectorized(weight_matrix, batch, delta, p, R, k,
                               one_over_scale, activation_function):
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
    k
        The rank of the hidden neuron whose synapses will undergo anti-hebbian
        learning.
    one_over_scale
        One over the time scale of learning.
    activation_function:
        The activation function of the hidden neurons.
    Return
    -----
    update
        ndarray, same shape as weight_matrix.
    """
    batch_update = np.zeros(weight_matrix.shape)

    (indexes_hebbian, indexes_anti) = ranker(batch, weight_matrix,
                                             activation_function, k, p)
    for i in range(len(batch)): #  If there's a better way, i haven't found it.

        j = indexes_hebbian[i]
        if j==61:
            pass #breakpoint()
        weight_vector_1 = weight_matrix[j]
        input_vector = batch[i]
        batch_update[j] += plasticity_rule(weight_vector_1, input_vector, 1, p,
                                           R, one_over_scale)

        j2 = indexes_anti[i]
        weight_vector_2 = weight_matrix[j2]
        batch_update[j2] += plasticity_rule(weight_vector_2, input_vector,
                                            -delta, p, R, one_over_scale)
    return batch_update


# %% defining the class


class CHUNeuralNetwork(TransformerMixin):
    """Extract features from data using a biology-inspired algorithm.

    Competing Hidden Units Neural Network. A 2-layers neural network
    that implements competition between patterns, learning unsupervised. The
    data transformed can then be used with a second, layer, supervised or not.
    See the article referenced in the notes for a more exhaustive explanation.

    Notes
    -----
        The name conventions for the variable is the same used in the article,
        when possible.
        As this network is composed of only two layers, the hidden neurons
        aren't actually hidden, but will be in practice as this network is
        meant to be used in conjunction with at least another layer.

    References
    ----------
        doi: 10.1073/pnas.1820458116
    """

    def __init__(self):
        pass

    def transform(self, X, activation_function=relu):
        """Transform the data."""
        return hidden_neurons_func(X, self.weight_matrix, activation_function)

    def fit(self, X, n_hiddens, delta=0.4, p=3, R=1, scale=1, k=7,
                 activation_function=relu, batch_size=None, epoch = None,
                 epochs = None):
        """Fit the weigths to the data.

        Intialize the matrix of weights, the put the data in minibatches and
        update the matrix for each minibatch.

        Parameters
        ----------
        X
            The data to fit. Shape: (sample, feature).
        n_hiddens:
            the number of hidden neurons
        delta:
            Relative strenght of anti-hebbian learning.
        p:
            Exponent of the lebesgue norm used (see product function).
        R:
            Radius of the sphere on wich the weights will converge.
        scale:
            The learning rate.
        k:
            The k-th most activated hidden neuron will undergo anti-hebbian
            learning.
        activation_function:
            The activation function of the hidden neurons.
        batch_size
            Number of elements in a batch.

        Return
        ------
        CHUNeuralNetwork
            The network itself.
        """
        if batch_size is None:
            batch_size = len(X)
        if not hasattr(self, "weight_matrix"): #  TODO ask: is it the correct way?
            dims = (n_hiddens, len(X[0]))
            self.weight_matrix = np.random.normal(0, 1/sqrt(n_hiddens), dims)
            # The weights are initialized with a gaussian distribution.

        update = np.zeros(self.weight_matrix.shape)
        for batch in batchize(X, batch_size):

            batch_update = plasticity_rule_vectorized(self.weight_matrix,
                                                      batch, delta, p, R,
                                                      k,
                                                      1/scale,
                                                      activation_function)
            update += batch_update
        scaled_update = scale_update(update, epoch, epochs)
        print(scaled_update[10][10])
        self.weight_matrix += scale_update(update, epoch, epochs)
        return self

    def fit_transform(self, X, batch_size=2):
        """Fit the data, then transform it."""
        return self.fit(X, batch_size).transform(X)
