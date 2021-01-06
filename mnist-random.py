"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
import pandas as pd
import CHUNeuralNetwork as chu
from time import time
import utilities as utils
#np.random.seed(2024)

(X_train, y_train, X_test, y_test) = utils.mnist_loader()
batch_size=100


# %% fit the data

layer1 = chu.CHUNeuralNetwork()
epochs=1


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
    if epoch%50 == 49:
        utils.image_representation(layer1.weight_matrix, 2, epoch,
                                   heatmap=True, pnorms=True,
                                   ravel=False)
print(time()-start)

# %% saving the results
    
data = pd.DataFrame(layer1.weight_matrix.copy())
data.to_hdf('matrices', key='random')
    
    # %% second layer
    
score = utils.score(X_train, y_train, X_test, y_test, layer1, (chu.activ, 4.5))
