

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import CHUNeuralNetwork as chu

np.random.seed(1024)
rng = np.random.default_rng(1024)


# %% loading and splitting the MNIST dataset

if not exists('./database_file'):
    mnist, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    mnist_dataframe = pd.DataFrame(mnist)
    mnist_dataframe.to_hdf("database_file", key="key")

X_train = np.array(pd.read_hdf("database_file"))/255.


# %%
Ns = len(X_train)
N = len(X_train[0])
eps0=0.02    # learning rate
Kx=10 # draw parameter
Ky=10 # draw parameter
hid=Kx*Ky    # number of hidden units that are displayed in Ky by Kx array
sigma=1.0 # init weight standard deviation
Nep=120      # number of epochs
Num=99      # size of the minibatch
prec=1e-30 # safety nonzero division parameter
delta=0.4    # Strength of the anti-hebbian learning
p=2.0        # Lebesgue norm of the weights
k=2          # ranking parameter, must be integer that is bigger or equal than 2

# %%


synapses = np.random.normal(0, sigma, (hid, N)) # init weights

for nep in range(Nep):
    rng.shuffle(X_train)
    for batch in chu.batchize(X_train, Num):
        inputs = batch.T
        my_ds = chu.plasticity_rule_vectorized(synapses, inputs.T, delta, p, 1, k,
                                   1, activation_function=None)
    
        update = chu.scale_update(my_ds, nep, Nep, learn_rate=0.02)
        
        synapses += update
    print(nep)


# %% image representation

def draw_weights(synapses, Kx, Ky):
    yy=0
    HM=np.zeros((28*Ky,28*Kx))
    for y in range(Kx):
        for x in range(Ky):
            HM[y*28:(y+1)*28,x*28:(x+1)*28]=synapses[yy,:].reshape(28,28)
            yy += 1
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    fig.canvas.draw()



fig=plt.figure(figsize=(12.9,10))
draw_weights(synapses, 10, 10)

