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

if not exists('./database_7000'):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X_train, X_test_7000, y1, y2 = train_test_split(X, y, test_size=0.1)
    database_7000 = open('database_7000', 'wb')
    pickle.dump(X_test_7000, database_7000)
    database_7000.close()

# %% doing the thing

database_7000 = open('./database_7000', 'rb')
X_test = pickle.load(database_7000)
database_7000.close()
# note to self: remember to change the number on all three lines

layer1 = chu.CHUNeuralNetwork(784)
fitted = layer1.fit(X_test[:432])  # the problem is at 432

transformed = fitted.transform(X_test[0])


# %% image representation


for i in range(10):
    synapsys, axsyn = plt.subplots()
    image = np.reshape(fitted.weight_matrix[i+1492
                                            ], (28, 28))
    axsyn = plt.imshow(image)
    plt.savefig("{}".format(i))


