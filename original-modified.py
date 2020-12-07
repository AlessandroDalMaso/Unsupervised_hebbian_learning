

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import CHUNeuralNetwork as chu

np.random.seed(1024)


M = np.array(pd.read_hdf("database_file"))/255.

N=784 # no. of input features
Ns = len(M)

# %%

def draw_weights(synapses, Kx, Ky):
    yy=0
    HM=np.zeros((28*Ky,28*Kx))
    for y in range(Ky):
        for x in range(Kx):
            HM[y*28:(y+1)*28,x*28:(x+1)*28]=synapses[yy,:].reshape(28,28)
            yy += 1
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    fig.canvas.draw()

# %%


eps0=0.02    # learning rate
Kx=10 # draw parameter
Ky=10 # draw parameter
hid=Kx*Ky    # number of hidden units that are displayed in Ky by Kx array
sigma=1.0 # init weight standard deviation
Nep=20      # number of epochs
Num=50      # size of the minibatch
prec=1e-30 # safety nonzero division parameter
delta=0.4    # Strength of the anti-hebbian learning
p=2.0        # Lebesgue norm of the weights
k=2          # ranking parameter, must be integer that is bigger or equal than 2

# %%

fig=plt.figure(figsize=(12.9,10))

synapses = np.random.normal(0, sigma, (hid, N)) # init weights
for nep in range(Nep): # epoch for cycle

    M=M[np.random.permutation(Ns),:] # random permutation of data
    for i in range(Ns//Num): # batch for cycle
        inputs=np.transpose(M[i*Num:(i+1)*Num,:]) # transposed
        """
        sig=np.sign(synapses)
        tot_input = chu.hidden_neurons_func_2(inputs.T, synapses, p).T # (j,i)
        #tot_input=np.dot(sig*np.absolute(synapses)**(p-1),inputs)
        # product? but also activation?
        # (100,784) dot (784,99) -> (100,99)

        #y=np.argsort(tot_input, axis=0)  # argsort... of product?

        #yl=np.zeros((hid,Num))  # g (j,i)
        #yl[y[hid-1],np.arange(Num)]=1.0
        #yl[y[hid-k],np.arange(Num)]=-delta
        (h, a) = chu.ranker(inputs.T, synapses, None, k, p)
        mine = np.zeros((Num, hid)) # (i,j)
        mine[np.arange(Num), h]=1.
        mine[np.arange(Num), a]=-0.4
        my_yl = mine.T
        
        ds1 = np.dot(my_yl, np.transpose(inputs)) # minuend
        xx=np.sum(np.multiply(my_yl,tot_input),1) # g dot <w,v>
        ds2 = np.multiply(np.tile(xx.reshape(xx.shape[0],1),(1,N)),synapses)
        # g dot <w,v> dot w
        ds = ds1 - ds2 # eq. 3
        """
        my_ds = chu.plasticity_rule_vectorized(synapses, inputs.T, delta, p, 1, k,
                               1, activation_function=None)

        update = chu.scale_update(my_ds, nep, Nep, learn_rate=0.02)
        
        synapses += update
    print(nep)
        
draw_weights(synapses, Kx, Ky)

