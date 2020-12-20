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

(X_train, y_train, X_test, y_test) = utils.mnist_loader(10000)

layer1 = chu.CHUNeuralNetwork()
half = 50
epochs=10

X = X_train.reshape((10, 9, len(X_train)//(10*9*50), 50, 784))
# (figure, 2nd figure, half-batch, sample, single pixel)

X_list = []
for figure in X:
    listt = []
    for halfbatch in figure:
        listt.append(halfbatch)
    X_list.append(listt)
# a list of lists of half batches, ready to be coupled

batches = []

for i in range(10):
    for j in range(i):
        batches.append(np.concatenate((X_list[i][0],X_list[j][0]), axis=1))
        X_list[i].remove(X_list[i][0])
        X_list[j].remove(X_list[j][0])