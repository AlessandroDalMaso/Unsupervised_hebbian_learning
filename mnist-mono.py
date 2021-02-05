import numpy as np
import pandas as pd
import utilities as utils
import CHUNeuralNetwork as chu
from time import time
from sklearn.ensemble import RandomForestClassifier
np.random.seed(6948)

(X_train, y_train, X_test, y_test) = utils.mnist_loader()
# loads 4500 samples for each figure, ordered in the array.

batch_size=100 # always a dividend of 4500
X = X_train.reshape((10, 4500, len(X_train[0])))
layer1 = chu.CHUNeuralNetwork()
epochs=160

start = time()
for epoch in range(epochs):
        X = X[:, np.random.permutation(4500)]
        # scramble each of the 10 figures
        batches = X.reshape((len(X_train)//batch_size, batch_size, 784))
        batches = batches[np.random.permutation(len(batches))]
        # then scramble the monotype batches
        for batch in batches:
            layer1 = layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0, p=2,
                                         R=1, scale=1, k=2, learn_rate=0.02,
                                         sigma=1, epoch=epoch, epochs=epochs)

utils.image_representation(layer1.weight_matrix, 2, epoch, heatmap=True,
                           pnorms=True, ravel=False)


print(time()-start)



# %% saving the results
    
data = pd.DataFrame(layer1.weight_matrix.copy())
data.to_hdf('results/matrices', key='monotype')

# %% second layer


score = utils.score(X_train, y_train, X_test, y_test, layer1, (chu.activ, 4.5))
print(score)

