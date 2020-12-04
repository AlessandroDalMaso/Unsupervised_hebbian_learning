"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import CHUNeuralNetwork as chu

np.random.seed(1024)


# %% loading and splitting the MNIST dataset

if not exists('./database_file'):
    mnist, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    mnist_train, mnist_test, y1, y2 = train_test_split(mnist, y,
                                                       test_size=0.28571)
    mnist_train_dataframe = pd.DataFrame(mnist_train)
    mnist_train_dataframe.to_hdf("database_file", key="key")

database = np.array(pd.read_hdf("database_file"))/255.


# %% Setting up the data

rng = np.random.default_rng()
rng.shuffle(database)
X_train = database.copy()

# %% fit and transform

layer1 = chu.CHUNeuralNetwork()
layer1 = layer1.fit( X=X_train, epochs=2, n_hiddens=100)
transformed = layer1.transform(X_train[0])

# %% image representation

def put_in_shape(matrix, rows, columns, indexes):
    """represent some weights"""
    counter = 0
    image=np.zeros((28*rows, 28*columns))
    for y in range(rows):
        for x in range(columns):
            shape = (28, 28)
            subimage = np.reshape(matrix[indexes[counter]], shape)
            image[y*28:(y+1)*28, x*28:(x+1)*28] = subimage
            counter += 1
    return image

indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]    

image = put_in_shape(layer1.weight_matrix.copy(), 3, 4, indexes)


vmax = np.amax(np.abs(image))

im, ax = plt.subplots()
ax = plt.imshow(image, cmap='bwr', vmax = vmax, vmin=-vmax)
plt.colorbar()
plt.savefig("image")

im2, ax2 = plt.subplots()
ax2 = plt.plot(chu.norms(layer1.weight_matrix, 3))
