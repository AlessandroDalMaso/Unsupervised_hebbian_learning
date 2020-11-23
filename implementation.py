"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import time
import CHUNeuralNetwork as chu

np.random.seed(1024)


# %% loading and splitting the MNIST dataset

if not exists('./database_file'):
    mnist, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    mnist_train, mnist_test, y1, y2 = train_test_split(mnist, y,
                                                       test_size=0.28571)
    mnist_train_dataframe = pd.DataFrame(mnist_train)
    mnist_train_dataframe.to_hdf("database_file", key="key")

# %% toy model testing
"""
start = time.time()
a = np.random.rand(30)
a = np.reshape(a, (3, 10))
toy_network = chu.CHUNeuralNetwork(n_hiddens=10)
toy_network.fit(a)
print(time.time()-start)
"""
# %% MNIST database testing

X_train = np.array(pd.read_hdf("database_file"))/225.

layer1 = chu.CHUNeuralNetwork()
start_time = time.time()
rng = np.random.default_rng()
for i in range(0, 4):
    rng.shuffle(X_train)
    layer1 = layer1.fit(X_train, batch_size=50000)
print("--- %s seconds ---" % (time.time() - start_time))
transformed = layer1.transform(X_train[0])

# %% image representation

for i in range(10):
    synapsys, axsyn = plt.subplots()
    image = np.reshape(layer1.weight_matrix[i], (28, 28))
    axsyn = plt.imshow(image, cmap='bwr')
    plt.savefig("{}".format(i))
