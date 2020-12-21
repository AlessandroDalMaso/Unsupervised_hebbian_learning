import numpy as np
import pandas as pd
import utilities as utils
import CHUNeuralNetwork as chu
from time import time
import random
from sklearn.ensemble import RandomForestClassifier
np.random.seed(1024)
random.seed(0)

(X_train, y_train, X_test, y_test) = utils.mnist_loader(test_size=0.16)
# loads 5000 samples for each figure, ordered in the array.

batch_size=100
X = X_train.copy()
batches = []
for i in np.arange(0,len(X), batch_size):
    batches.append(X[i:i+batch_size])


layer1 = chu.CHUNeuralNetwork()
epochs=10

start = time()
for epoch in range(epochs):
    random.shuffle(batches)
    for batch in batches:
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
# no transform: 0.97

