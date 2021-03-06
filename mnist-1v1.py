"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
import pandas as pd
import CHUNeuralNetwork as chu
from time import time
import utilities as utils
np.random.seed(1024)


(X_train, y_train, X_test, y_test) = utils.mnist_loader()

layer1 = chu.CHUNeuralNetwork()
half = 50

X = X_train.reshape((10, len(X_train)//10, 784))
# (figure, sample, single pixel)







epochs=160
layer1 = chu.CHUNeuralNetwork()

start = time()
for epoch in range(epochs):
    X = X[:, np.random.permutation(len(X[0]))]
    #shuffle between same figures
    Y = X.reshape((10, len(X_train)//(10*half), half, 784))
    # (figure, half-batch, sample, single pixel)
    
    list1 = []
    for figure in Y:
        list1.append(list(figure))
    # make it a list for removal
    
    batches = []
    for i in range(10):
        # for each figure...
        for j in range(i):
            #... for each figure combination (i,j)...
            for k in range(len(X_train)//(10*9*half)):
                #...take a certain number of images for both figures...
                batches.append(np.concatenate((list1[i][0],list1[j][0])))
                # ...make a batch out of them...
                list1[i].remove(list1[i][0])
                list1[j].remove(list1[j][0])
                # ...and remove them from the list!

    for b in batches:
        layer1 = layer1.fit_single_batch(batch=b, n_hiddens=100, delta=0.4, p=2,
                                         R=1, scale=1, k=2, learn_rate=0.02,
                                         sigma=10, hh=1, epoch=epoch, epochs=epochs)
            
    print(epoch)
utils.image_representation(layer1.weight_matrix, 2, epoch=epoch, heatmap=True,
                                       pnorms=True, ravel=False)

#print(time()-start)



# %% saving the results
    
data = pd.DataFrame(layer1.weight_matrix.copy())
data.to_hdf('results/matrices', key='_1v1')

# %% second layer


score = utils.score(X_train, y_train, X_test, y_test, layer1, (chu.activ, 4.5))
print(score)

    