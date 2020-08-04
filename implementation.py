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

np.random.seed(12345)

# %% loading and splitting the MNIST dataset

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X_train, X_test, y1, y2 = train_test_split(X, y, test_size = 0.001)


# %% doing the thing
"""
layer1 = chu.CHUNeuralNetwork(784)
fitted = layer1.fit(X_test)
transformed = fitted.transform([X_test[0]])
"""
toy = chu.CHUNeuralNetwork(4)
a = np.random.rand(100, 4)
fitted_toy = toy.fit(a)
tranformed_toy = toy.transform(a)



# %% image representation
"""
image = np.reshape(fitted.weight_matrix[1], (28, 28))
fig, ax = plt.subplots()
ax = plt.imshow(image)
plt.savefig("fig.png")
ori = np.reshape(X_test[0], (28, 28))
fig2, ax2 = plt.subplots()
ax2 = ori
ax2 = plt.imshow(ax2)
plt.savefig("fig2.png")
"""





