"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import CHUNeuralNetwork as chu

np.random.seed(1024)


# %% loading and splitting the MNIST dataset

if not exists('./database_file'):
    mnist, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    mnist_train, mnist_test, y1, y2 = train_test_split(mnist, y,
                                                       test_size=0.28571)
    mnist_train_dataframe = pd.DataFrame(mnist_train)
    mnist_train_dataframe.to_hdf("database_file", key="key")

X_train = np.array(pd.read_hdf("database_file"))/255.

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

n_hiddens = 100

layer1 = chu.CHUNeuralNetwork(n_hiddens)
rng = np.random.default_rng()

# %%

x=0
for i in range(1):
    x+=1
    rng.shuffle(X_train)
    layer1 = layer1.fit(X_train, batch_size=50000)
    print(x, " epochs have been processed")
transformed = layer1.transform(X_train[0])

# %% image representation

counter = 0
image=np.zeros((28*3, 28*4))
matrix = layer1.weight_matrix.copy()
for y in range(3):
    for x in range(4):
        image[y*28:(y+1)*28, x*28:(x+1)*28] = np.reshape(matrix[counter], (28,28))
        counter += 1

vmax = np.amax(np.abs(image))

im, ax = plt.subplots()
ax = plt.imshow(image, cmap='bwr', vmax = vmax, vmin=-vmax)
plt.colorbar()
plt.savefig("image")




