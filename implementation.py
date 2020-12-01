"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import CHUNeuralNetwork as chu
from math import sqrt

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

layer1 = chu.CHUNeuralNetwork(n_hiddens=100, scale=10)

epochs = 1
rng = np.random.default_rng()
rng.shuffle(database)
X_train = database.copy()
for i in range(epochs-1):
    rng.shuffle(database)
    X_train = np.concatenate((X_train, database), axis=0)

# %% fit and transform

layer1 = layer1.fit(X_train, 50000)
transformed = layer1.transform(X_train[0])

# %% image representation

def put_in_shape(matrix, rows, columns, indexes):
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

image = put_in_shape(layer1.weight_matrix.copy(), 3, 4, 28, 28, indexes)


vmax = np.amax(np.abs(image))

im, ax = plt.subplots()
ax = plt.imshow(image, cmap='bwr', vmax = vmax, vmin=-vmax)
plt.colorbar()
plt.savefig("image")
