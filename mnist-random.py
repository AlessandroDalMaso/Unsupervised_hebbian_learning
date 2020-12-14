"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import CHUNeuralNetwork as chu
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

np.random.seed(1024)


# %% loading, splitting  and normalizing the MNIST dataset

if not exists('./data/mnist'):
    bunch = fetch_openml('mnist_784', version=1, as_frame=True)
    bunch.frame.to_hdf('data/mnist', key='key', format='table')
database = pd.read_hdf('data/mnist', key='key')
data = database.drop('class', axis=1)/255
target = database['class']



# %%


layer1 = chu.CHUNeuralNetwork()
epochs=10
X_train = np.array(data)


# %% fit the data


start = time()
for epoch in range(epochs):
    X_train=X_train[np.random.permutation(len(X_train)),:]
    for batch in chu.batchize(X_train, batch_size=99):

        layer1 = layer1.fit( batch=batch, n_hiddens=100, delta=0.4, p=2, R=1,
                            scale=1, k=2, learn_rate=0.02, sigma=1,
                            activation_function=chu.relu, batch_size=99,
                            epoch=epoch, epochs=epochs)
    print(epoch)

print(time()-start)




# %% image representation

matrix = layer1.weight_matrix.copy()

image = chu.put_in_shape(matrix, 10, 10)
vmax = np.amax(np.abs(image))
im, ax = plt.subplots()
ax = plt.imshow(image, cmap='bwr', vmax = vmax, vmin=-vmax)
plt.colorbar()
plt.savefig("images/mnist-random/weights_heatmap")

im2, ax2 = plt.subplots()
ax2 = plt.plot(chu.norms(matrix, 3))
plt.savefig("images/mnist-random/p-norms")

im3, ax3 = plt.subplots()
ax3 = plt.plot(np.ravel(matrix))
plt.savefig("images/mnist-random/weights_unraveled")


# %% second layer


forest = RandomForestClassifier()
delta = 0.4

fit_params = {'delta': 0.4}

pipeline = make_pipeline(layer1, RandomForestClassifier())
scores = cross_val_score(pipeline, data, target, CHUNeuralNetwork__delta=delta)
