"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
import CHUNeuralNetwork as chu
from time import time
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import utilities as utils
import random
np.random.seed(1024)
random.seed(0)

(X_train, y_train, X_test, y_test) = utils.mnist_loader(test_size=10000)
batch_size=100


# %% fit the data

layer1 = chu.CHUNeuralNetwork()
epochs=160


start = time()

for epoch in range(epochs):
    X = X_train[np.random.permutation(len(X_train))]
    batches = X.reshape((45000//batch_size, batch_size, 784))
    for batch in batches:
        layer1 = layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0.4,
                                         p=3,
                                         R=1, scale=1, k=5, learn_rate=0.01,
                                         sigma=6, epoch=epoch, epochs=epochs)
    print(epoch)
print(time()-start)

utils.image_representation(layer1.weight_matrix, 2)


# %% second layer


transformed_train = layer1.transform(X_train, chu.activ, 4.5)
transformed_test = layer1.transform(X_test, chu.activ, 4.5)

forest1 = RandomForestClassifier()

start=time()
forest1.fit(transformed_train, y_train)
print(time()-start)

score1 = forest1.score(transformed_test, y_test)
# my score: 0.94
# no transform: 97