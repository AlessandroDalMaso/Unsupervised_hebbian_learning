import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py
from os.path import exists
import time

import CHUNeuralNetwork as chu

np.random.seed(1024)


# %% loading and splitting the MNIST dataset

if not exists('./database_7000.hdf5'):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X_train, X_test_7000, y1, y2 = train_test_split(X, y, test_size=0.1)
    database_7000 = h5py.File("database_7000.hdf5", "w")
    database_7000.create_dataset("data", data=X_train)
    database_7000.close()

# %% toy model testing
start = time.time()
a = np.random.rand(30)
a = np.reshape(a, (3, 10))
toy_network = chu.CHUNeuralNetwork(n_hiddens=10)
toy_network.fit(a)
print(time.time()-start)

# %% MNIST database testing
"""
database_7000 = h5py.File("database_7000.hdf5", "r")
X_test = database_7000["data"]
database_7000.close()

layer1 = chu.CHUNeuralNetwork()
start_time = time.time()
layer1 = layer1.fit(X_test[:1])  # the problem is at 432
print("--- %s seconds ---" % (time.time() - start_time))
transformed = layer1.transform(X_test[0])

# %% image representation

for i in range(2):
    synapsys, axsyn = plt.subplots()
    image = np.reshape(layer1.weight_matrix[i], (28, 28))
    axsyn = plt.imshow(image, cmap='bwr')
    plt.savefig("{}".format(i))
"""
