"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
import matplotlib.pyplot as plt
import CHUNeuralNetwork as chu
from time import time
import utilities as utils
import random
np.random.seed(1024)
random(0)

(X_train, y_train, X_test, y_test) = utils.mnist_loader(10000)

layer1 = chu.CHUNeuralNetwork()
half = 50


X = X_train.reshape((10, 9, len(X_train)//(10*9*half), half, 784))
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
        chunk = np.concatenate((X_list[i][0],X_list[j][0]), axis=1)
        X_list[i].remove(X_list[i][0])
        X_list[j].remove(X_list[j][0])
        for batch in chunk:
            batches.append(batch)
# a list of lists of batches 1 figure vs. another

strart = time()


# %%

epochs=10
layer1 = chu.CHUNeuralNetwork()

start = time()
for epoch in range(epochs):
    random.shuffle(batches)
    for b in batches:
        layer1 = layer1.fit_single_batch(batch=b, n_hiddens=100, delta=0.4, p=2,
                                         R=1, scale=1, k=2, learn_rate=0.02,
                                         sigma=1, epoch=epoch, epochs=epochs)
    print(epoch)

print(time()-start)

utils.image_representation(layer1.weight_matrix)

# %%

    