"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
import pandas as pd
import CHUNeuralNetwork as chu
from time import time
import utilities as utils
np.random.seed(3024)

(X_train, y_train, X_test, y_test) = utils.mnist_loader()
batch_size=100


# %% fit the data

layer1 = chu.CHUNeuralNetwork()
epochs=100


start = time()

for epoch in range(epochs):
    X = X_train[np.random.permutation(len(X_train))]
    batches = X.reshape((45000//batch_size, batch_size, 784))
    for batch in batches:
        layer1 = layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0,
                                         p=2, R=1, scale=1, k=2,
                                         learn_rate=0.1, sigma=10,
                                         epoch=epoch, epochs=epochs)
    print(epoch)
    if epoch<10:
        utils.image_representation(layer1.weight_matrix, 2, epoch,
                                   heatmap=True, pnorms=False,
                                   ravel=False)
utils.image_representation(layer1.weight_matrix, 2, epoch,
                                   heatmap=True, pnorms=False,
                                   ravel=False)
#print(time()-start)

# %% saving the results
    
data = pd.DataFrame(layer1.weight_matrix.copy())
data.to_hdf('results/matrices', key='random')
    
    # %% second layer
    
utils.score(X_train, y_train, X_test, y_test, layer1, (chu.activ, 4.5))
