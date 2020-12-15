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

(X_train, y_train, X_test, y_test) = utils.mnist_loader()

X_0 = X_train[np.where(y_train == '0')]
X_1 = X_train[np.where(y_train == '1')]
X_2 = X_train[np.where(y_train == '2')]
X_3 = X_train[np.where(y_train == '3')]
X_4 = X_train[np.where(y_train == '4')]
X_5 = X_train[np.where(y_train == '5')]
X_6 = X_train[np.where(y_train == '6')]
X_7 = X_train[np.where(y_train == '7')]
X_8 = X_train[np.where(y_train == '8')]
X_9 = X_train[np.where(y_train == '9')]

layer1 = chu.CHUNeuralNetwork()
epochs=160
start = time()
for epoch in range(epochs):
    X = X_train[np.random.permutation(len(X_train)),:]
    for batch in utils.batchize(X, batch_size=99):

        layer1 = layer1.fit( batch=batch, n_hiddens=100, delta=0.4, p=2, R=1,
                            scale=1, k=2, learn_rate=0.02, sigma=1,
                            activation_function=chu.relu, batch_size=99,
                            epoch=epoch, epochs=epochs)
    print(epoch)

print(time()-start)

utils.image_representation(layer1.weight_matrix)

