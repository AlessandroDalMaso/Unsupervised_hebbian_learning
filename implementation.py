"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import CHUNeuralNetwork as chu
from time import time

np.random.seed(1024)


# %% loading and splitting the MNIST dataset

if not exists('./mnist'):
    bunch = fetch_openml('mnist_784', version=1, as_frame=True)
    bunch.frame.to_hdf('mnist', key='key', format='table')
database = pd.read_hdf('mnist', key='key')
unlabeled = database.drop('class', axis=1)
X_train = np.array(unlabeled)/255.






layer1 = chu.CHUNeuralNetwork()
epochs = 10








# %% fit and transform



start = time()
for epoch in range(epochs):
    X_train=X_train[np.random.permutation(len(X_train)),:]
    for batch in chu.batchize(X_train, batch_size=99):
        layer1 = layer1.fit( batch=batch, n_hiddens=100, delta=0.4, p=2, R=1, scale=1, k=2, learn_rate=0.02, activation_function=chu.relu, batch_size=99, epoch=epoch, epochs=epochs)
    print(epoch)
print(time()-start)







# %% image representation



image = chu.put_in_shape(layer1.weight_matrix.copy(), 10, 10)
vmax = np.amax(np.abs(image))
im, ax = plt.subplots()
ax = plt.imshow(image, cmap='bwr', vmax = vmax, vmin=-vmax)
plt.colorbar()
plt.savefig("image")

im2, ax2 = plt.subplots()
ax2 = plt.plot(chu.norms(layer1.weight_matrix, 3))

im3, ax3 = plt.subplots()
ax3 = plt.plot(np.ravel(layer1.weight_matrix))
