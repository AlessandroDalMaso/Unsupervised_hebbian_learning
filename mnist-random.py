"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
import matplotlib.pyplot as plt
import CHUNeuralNetwork as chu
from time import time
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import utilities as utils

np.random.seed(1024)
(X_train, y_train, X_test, y_test) = utils.mnist_loader()




# %% fit the data

layer1 = chu.CHUNeuralNetwork()
epochs=10
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




# %% image representation

matrix = layer1.weight_matrix.copy() # type?????

image = utils.put_in_shape(matrix, 10, 10)
vmax = np.amax(np.abs(image))
im, ax = plt.subplots()
ax = plt.imshow(image, cmap='bwr', vmax = vmax, vmin=-vmax)
plt.colorbar()
plt.savefig("images/mnist-random/weights_heatmap")

im2, ax2 = plt.subplots()
ax2 = plt.plot(utils.norms(matrix, 3))
plt.savefig("images/mnist-random/p-norms")

im3, ax3 = plt.subplots()
ax3 = plt.plot(np.ravel(matrix))
plt.savefig("images/mnist-random/weights_unraveled")


# %% second layer

transformed_train = X_train @ layer1.weight_matrix.T
transformed_test = X_test @ layer1.weight_matrix.T

forest1 = RandomForestClassifier()
forest2 = RandomForestClassifier()

start=time()
forest1.fit(transformed_train, y_train)
forest2.fit(X_train, y_train)
print(time()-start)

score1 = forest1.score(transformed_test, y_test) # 0.986
score2 = forest2.score(X_test, y_test) # 0.939



