import numpy as np
import pandas as pd
import utilities as utils
import CHUNeuralNetwork as chu
np.random.seed(1024)

(X_train, y_train, X_test, y_test) = utils.mnist_loader(test_size=0.16)
# loads 5000 samples for each figure, ordered in the array.

layer1 = chu.CHUNeuralNetwork()
epochs_per_figure=1
batch_size=99

for n in range(10):
    indexes = np.random.permutation(5000) + n * 5000
    data = X_train[indexes]
    for epoch in range(epochs_per_figure):
        for batch in utils.batchize(data, batch_size):
            layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0.4, p=2, R=1,
                                    scale=1, k=2, learn_rate=0.02, sigma=1,
                                    batch_size=99, epoch=epoch,
                                    epochs=epochs_per_figure)
    print(n)
    
