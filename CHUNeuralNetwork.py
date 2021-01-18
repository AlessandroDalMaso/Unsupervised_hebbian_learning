"""A biology-inspired data transformer."""

import numpy as np

# %% defining external equations


def scale_update(update, epoch, epochs, learn_rate):
    """scale the update to avoid overshooting"""
    max_norm = np.amax(np.abs(update))
    #avg = np.average(np.abs(update))
    esp = (1-epoch/epochs)
    return learn_rate*esp*update/max_norm


def activ(currents, n):
    """Is the default activation function."""
    return np.where(currents < 0, 0, np.sign(currents) * np.abs(currents) ** n)
    # no need to worry about complex numbers


def product(weight_matrix, batch, p):
    """equation [2] of the refernce article, vectorized."""
    return batch @ (np.sign(weight_matrix) * np.abs(weight_matrix) ** (p-1)).T


def plasticity_rule(weight_vector, input_vector, product_result, g, p, R,
                    one_over_scale):
    """Equation [3] of the original article."""
    minuend = R ** p * input_vector
    subtrahend = product_result * weight_vector
    return g * (minuend - subtrahend) * one_over_scale
    


def plasticity_rule_vectorized(weight_matrix, batch, delta, p, R, k,
                               one_over_scale):
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
    Return
    -----
    update: ndarray
        same shape as weight_matrix.
    """
    product_result = product(weight_matrix, batch, p)
    sorting = np.argsort(product_result) # batch @ weight_matrix.T?
    update = np.zeros(weight_matrix.shape)
    for i in range(len(batch)):
        h = sorting[i,-1]
        a = sorting[i,-k]

        update[h] += plasticity_rule(weight_matrix[h], batch[i],
                                     product_result[i,h], 1, p, R,
                                     one_over_scale)
        if delta is not 0:
            update[a] += plasticity_rule(weight_matrix[a], batch[i],
                                         product_result[i,a], -delta, p, R,
                                         one_over_scale)
    return update

# %% defining the class


class CHUNeuralNetwork():
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

    def transform(self, X, activation_function=activ, *args):
        """Transform the data."""
        return activation_function(X @ self.weight_matrix.T, *args)
    # UUUUUUUUUUUUUUUUUUUUUUUU

    def fit_single_batch(self, batch, n_hiddens, delta, p, R, scale, k,
                         learn_rate, sigma, epoch, epochs):
        """Fit the weigths to a single batch.

        Intialize the matrix of weights, then update the matrix with the result
        of plasticity_rule_vectorized.

        Parameters
        ----------
        batch
            The data to fit. Shape: (sample, feature).
        n_hiddens:
            the number of hidden neurons.
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
        learn_rate:
            The update will be normalized so that its maximum (in absolute
            value) will be equal to this parameter.
        sigma:
            The standard deviation of the normal distribution from wich random
            weights initial values are drawn.
        epoch:
            The epoch number.
        epochs:
            the total number of epochs.

        Return
        ------
        CHUNeuralNetwork
            The network itself.
        """
        dims = (n_hiddens, len(batch[0]))
        if not hasattr(self, "weight_matrix"):  
            self.weight_matrix = np.random.normal(0, sigma, dims)
            #self.weight_matrix = np.abs(self.weight_matrix)
            #for i in range(len(self.weight_matrix)):
            #    norm = np.sum(np.abs(self.weight_matrix[i]) ** p)
            #    self.weight_matrix[i] = self.weight_matrix[i]/(norm ** (1/p))
            # The weights are initialized with a gaussian distribution.
            


        update = plasticity_rule_vectorized(self.weight_matrix,
                                            batch, delta, p, R, k, 1/scale,)

        scaled_update = scale_update(update, epoch, epochs, learn_rate)
        self.weight_matrix += scaled_update
        
        return self

    def fit(self, database, n_hiddens, delta, p, R, scale, k, learn_rate,
            sigma, batch_size, epochs):
        """Fit the weigths to the whole database.

        Intialize the matrix of weights, then put the data in minibatches, then
        fit the matrix of weights to each batch.

        Parameters
        ----------
        database
            The data to fit. Shape: (sample, feature).
        n_hiddens:
            the number of hidden neurons.
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
        learn_rate:
            The update will be normalized so that its maximum (in absolute
            value) will be equal to this parameter.
        sigma:
            The standard deviation of the normal distribution from wich random
            weights initial values are drawn.
        batch_size:
            The size of the batches to put samples in.
        epochs:
            the total number of epochs.

        Return
        ------
        CHUNeuralNetwork
            The network itself.
        """

        dims = (n_hiddens, len(database[0]))
        if not hasattr(self, "weight_matrix"):  
            self.weight_matrix = np.random.triangular(-sigma, 0, sigma, dims)
            # The weights are initialized with a gaussian distribution.

        for epoch in range(epochs):
            X = database[np.random.permutation(len(database))]
            for i in range(0, len(X), batch_size):
                batch=X[i:i+batch_size]
                self.fit_single_batch(batch, n_hiddens, delta, p, R, scale, k,
                         learn_rate, sigma, epoch, epochs)
            print(epoch)

        return self

    def fit_transform(self, X, n_hiddens, delta, p, R, scale, k, learn_rate,
                      sigma, activation_function, batch_size, epoch, epochs):
        """Fit the data, then transform it."""
        return self.fit(X, n_hiddens, delta, p, R, scale, k, learn_rate,
                 activation_function, batch_size, epoch,epochs).transform(X)
