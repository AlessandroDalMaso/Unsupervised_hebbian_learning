"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import CHUNeuralNetwork as chu

np.random.seed(1024)
rng = np.random.default_rng(1024)


# %% loading and splitting the MNIST dataset

if not exists('./database_file'):
    mnist, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    mnist_dataframe = pd.DataFrame(mnist)
    mnist_dataframe.to_hdf("database_file", key="key")

X_train = np.array(pd.read_hdf("database_file"))/255.


# %% fit and transform

layer1 = chu.CHUNeuralNetwork()
epochs = 2


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



for epoch in range(epochs):
    rng.shuffle(X_train)
    for batch in batchize(X_train, batch_size=99):
        layer1 = layer1.fit( batch=batch, n_hiddens=100, delta=0.4, p=2, R=1, scale=1, k=2, learn_rate=0.02, activation_function=chu.relu, batch_size=99, epoch=epoch, epochs=epochs)
    print(epoch)







# %% image representation

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

image = put_in_shape(layer1.weight_matrix.copy(), 10, 10)
vmax = np.amax(np.abs(image))
im, ax = plt.subplots()
ax = plt.imshow(image, cmap='bwr', vmax = vmax, vmin=-vmax)
plt.colorbar()
plt.savefig("image")

im2, ax2 = plt.subplots()
ax2 = plt.plot(chu.norms(layer1.weight_matrix, 3))

im3, ax3 = plt.subplots()
ax3 = plt.plot(np.ravel(layer1.weight_matrix))
