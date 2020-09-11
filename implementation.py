# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:17:42 2020

@author: Alessandro Dal Maso
"""


import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import CHUNeuralNetwork as chu
import pickle
from os.path import exists

np.random.seed(12345)

# %% loading and splitting the MNIST dataset

if not exists('./database_7'):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X_train, X_test_7, y1, y2 = train_test_split(X, y, test_size=0.0001)
    # current train set: 7.000 objective: 60.000
    database_7 = open('database_7', 'wb')
    pickle.dump(X_test_7, database_7)
    database_7.close()
if not exists('./database_70'):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X_train, X_test_70, y1, y2 = train_test_split(X, y, test_size=0.001)
    # current train set: 7.000 objective: 60.000
    database_70 = open('database_70', 'wb')
    pickle.dump(X_test_70, database_70)
    database_70.close()
if not exists('./database_700'):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X_train, X_test_700, y1, y2 = train_test_split(X, y, test_size=0.01)
    # current train set: 7.000 objective: 60.000
    database_700 = open('database_700', 'wb')
    pickle.dump(X_test_700, database_700)
    database_700.close()
if not exists('./database_7000'):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X_train, X_test_7000, y1, y2 = train_test_split(X, y, test_size=0.1)
    # current train set: 7.000 objective: 60.000
    database_7000 = open('database_7000', 'wb')
    pickle.dump(X_test_7000, database_7000)
    database_7000.close()

# %% doing the thing

test = open('./test', 'rb')
X_test = pickle.load(database_70)
database_70.close()

layer1 = chu.CHUNeuralNetwork(784)
fitted = layer1.fit(X_test)

transformed = fitted.transform(X_test[0])

"""
toy = chu.CHUNeuralNetwork(4)
a = np.random.rand(10, 4)
fitted_toy = toy.fit(a)
tranformed_toy = toy.transform(a)
"""
# %% image representation

fig, ax = plt.subplots()
image = np.reshape(transformed, (50, 40))
ax = plt.imshow(image)

for i in range(20):
    synapsys, axsyn = plt.subplots()
    image = np.reshape(fitted.weight_matrix[i], (28, 28))
    axsyn = plt.imshow(image)
    plt.savefig("{}".format(i))


