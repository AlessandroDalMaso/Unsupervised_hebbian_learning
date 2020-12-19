import numpy as np
import pandas as pd
import utilities as utils
import CHUNeuralNetwork as chu
from time import time
from sklearn.ensemble import RandomForestClassifier
np.random.seed(1024)

(X_train, y_train, X_test, y_test) = utils.mnist_loader(test_size=0.16)
# loads 5000 samples for each figure, ordered in the array.

layer1 = chu.CHUNeuralNetwork()
epochs_per_figure=1
batch_size=100
epochs=10

start = time()
for epoch in range(epochs):
    X = X_train.copy()
    indexes = np.random.permutation(len(X)//batch_size)*batch_size
    for i in indexes[np.random.permutation(len(indexes))]:
        batch = X[i:i+batch_size]
        layer1 = layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0.4, p=2,
                                         R=1, scale=1, k=2, learn_rate=0.02,
                                         sigma=1, epoch=epoch, epochs=epochs)
    print(epoch)

print(time()-start)

utils.image_representation(layer1.weight_matrix)

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

