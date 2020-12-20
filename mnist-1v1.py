"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import CHUNeuralNetwork as chu
from time import time
import utilities as utils
np.random.seed(1024)

(X_train, y_train, X_test, y_test) = utils.mnist_loader(0.16)

layer1 = chu.CHUNeuralNetwork()
half = 50
epochs=10

X = X_train.reshape((10, (4500//half), half, 784))

