# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:17:42 2020

@author: Alessandro Dal Maso
"""

import CHUNeuralNetwork as chu
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %% loading and splitting the MNIST dataset

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)


# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9999)

layer1 = chu.CHUNeuralNetwork(784)
transformed = layer1.transform(X_train)
fitted = layer1.fit(X_train)
#image = np.reshape(transformed, (3, 3))
#b = plt.imshow(image)
