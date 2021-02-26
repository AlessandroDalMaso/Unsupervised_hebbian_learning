"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
import CHUNeuralNetwork as chu
from time import time
import utilities as utils
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import pandas as pd
#np.random.seed(3024)

(X_train, y_train, X_test, y_test) = utils.mnist_loader()
batch_size=100

# %% execution

layer1 = chu.CHUNeuralNetwork()
epochs=100


start = time()

for epoch in range(epochs):
    X = X_train[np.random.permutation(len(X_train))]
    batches = X.reshape((45000//batch_size, batch_size, 784))
    for batch in batches:
        layer1 = layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0.4,
                                         p=2, R=1, scale=1, k=2,
                                          learn_rate=0.04, sigma=10, hh=1, aa=1,
                                         decay = 0, epoch=epoch, epochs=epochs)
    print(epoch)
utils.image_representation(layer1.weight_matrix, 2, epoch,
                                   heatmap=True, pnorms=True,
                                   ravel=False)
#print(time()-start)

# %% saving the results

data = pd.DataFrame(layer1.weight_matrix.copy())
data.to_hdf('results/matrices', key='random')

# %% second layer

#utils.score(X_train, y_train, X_test, y_test, layer1, (chu.activ, 4.5))
args=(1)
t_train = layer1.transform(X_train, chu.activ, args)
args = (1)
t_test = layer1.transform(X_test, chu.activ, args)
pip_perceptron = make_pipeline(StandardScaler(), SGDClassifier(loss='perceptron', max_iter=2000))
pip_perceptron.fit(utils.h_activation(t_train), y_train)
score5 = pip_perceptron.score(np.abs(t_test), y_test)
print('second layer transform score: ', score5)