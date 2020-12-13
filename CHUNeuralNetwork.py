"""A biology-inspired data transformer."""

from sklearn.base import TransformerMixin
import numpy as np
from scipy.integrate import solve_ivp

# %% defining external equaltions

def batchize(iterable, batch_size):
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
    for n in range(0, lenght, batch_size):
        yield iterable[n:min(n + batch_size, lenght)]

def put_in_shape(matrix, rows, columns, indexes=None):
    """represent some weights"""
    if indexes is None:
        indexes = range(len(matrix))
    counter = 0
    image=np.zeros((28*rows, 28*columns))
    for y in range(rows):
        for x in range(columns):
            shape = (28, 28)
            subimage = np.reshape(matrix[indexes[counter]], shape)
            image[y*28:(y+1)*28, x*28:(x+1)*28] = subimage
            counter += 1
    return image


def norms(matrix, p):
    """p-norms of vectors in a matrix."""
    return np.sum(np.abs(matrix) ** p, axis=1)





def scale_update(update, epoch, epochs, learn_rate):
    """scale the update like in the original code"""
    max_norm = np.amax(np.abs(update))
    esp = learn_rate*(1-epoch/epochs)
    return esp*update/max_norm


def relu(currents):
    """Is the default activation function."""
    return np.where(currents < 0, 0, currents)

def hidden_neurons_func(batch, weight_matrix, activation_function):
    """Calculate hidden neurons activations."""
    currents = batch @ weight_matrix.T
    #currents2 = np.einsum("ik,jk->ij", batch, weight_matrix)
    #currents3 = np.einsum("ik,kj->ij", batch, weight_matrix.T)
    # TODO explain the different behavior
    return activation_function(currents)


def hidden_neurons_func_2(batch, weight_matrix, p):
    """Calculate hidden neurons activations. like in the original code."""
    product = weight_matrix * np.abs(weight_matrix) ** (p - 1)
    return batch @ product.T
    # (i, k) @ (k, j) = (i, j)


def ranker(batch, weight_matrix, activation_function, k, p):
    """Return the indexes of the first and k-th most activated neurons."""
    hidden_neurons = hidden_neurons_func_2(batch, weight_matrix, p)
    sorting = np.argsort(hidden_neurons)
    return (sorting[:, -1], sorting[:, -k]) # dim i

def g(batch, weight_matrix, k, delta):
    """Return a learning activation function for each hidden neuron.
    
    
    """
    hiddens = batch @ weight_matrix.T
    sort = np.argsort(hiddens, axis=1)
    result = np.zeros(sort.shape)
    result[:,-1] = 1
    result[:,-k] = -delta
    return result


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
    sig = np.sign(weight_vector)
    product = sig * (np.abs(weight_vector) ** p-1) * input_vector
    return np.sum(product)

def product_v(weight_matrix, batch, p):
    return batch @ (np.sign(weight_matrix) * np.abs(weight_matrix) ** (p-1)).T


def plasticity_rule(weight_vector, input_vector, product_result, g, p, R, one_over_scale):
    """Equation [3] of the original article."""
    return g * (R ** p * input_vector - product_result * weight_vector
                ) * one_over_scale
    


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
    product_result = product_v(weight_matrix, batch, p)
    sorting = np.argsort(product_result)
    update = np.zeros(weight_matrix.shape)
    for i in range(len(batch)): #  alternative: add.at()
        h = sorting[i,-1]
        a = sorting[i,-k]

        update[h] += plasticity_rule(weight_matrix[h], batch[i],
                                     product_result[i,h], 1, p, R,
                                     one_over_scale)

        update[a] += plasticity_rule(weight_matrix[a], batch[i],
                                     product_result[i,a], -delta, p, R,
                                     one_over_scale)

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

    def fit(self, batch, n_hiddens, delta, p, R, scale, k, learn_rate,
                 activation_function, batch_size, epoch,
                 epochs):
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
        if not hasattr(self, "weight_matrix"): #  TODO ask: is it the correct way?
            dims = (n_hiddens, len(batch[0]))
            self.weight_matrix = np.random.normal(0, 1, dims) # TODO sigma
            # The weights are initialized with a gaussian distribution.


        update = plasticity_rule_vectorized(self.weight_matrix,
                                            batch, delta, p, R, k, 1/scale,
                                            activation_function)
        scaled_update = scale_update(update, epoch, epochs, learn_rate)
        self.weight_matrix += scaled_update
        return self

    def fit_transform(self, X, n_hiddens, delta, p, R, scale, k, learn_rate,
                 activation_function, batch_size, epoch,
                 epochs):
        """Fit the data, then transform it."""
        return self.fit(X, n_hiddens, delta, p, R, scale, k, learn_rate,
                 activation_function, batch_size, epoch,epochs).transform(X)
