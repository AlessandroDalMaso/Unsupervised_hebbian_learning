## 12/12/2020

### 1

con il mio codice ho fatto 160 epochs con batch a 100.
<details>

![](12-12-2020/1/2020-12-12-weights.png)

![](12-12-2020/1/2020-12-12-norms.png)

![](12-12-2020/1/2020-12-12-ravel.png)
</details>

## 15/12/2020

### 1
con il codice originale, 160 batch da 50 samples
<details>
![](15-12-2020/1/15-12-2020-original.png)
</details>

### 2

ho splittato mnist 25% di test e ho fittato una random forest, senza preprocessare. risultato: score = 0.97

### 3

ho fatto un fit con mnist 25% di test 160 epochs batch da 99 e ho fittato una random forest con i risultati. risultato: score = 0.943
<details>
![](15-12-2020/2/p-norms.png)

![](15-12-2020/2/weights_heatmap.png)

![](15-12-2020/2/weights_unraveled.png)
</details>

# 16/12/2020

## 1

codice: mio

splitting: random

batch: 160

splitting: 16%
<details>

```
layer1.fit_single_batch( batch=batch, n_hiddens=100, delta=0.4, p=2, R=1,
                            scale=1, k=2, learn_rate=0.02, sigma=1,
                            activation_function=chu.activ, batch_size=99,
                            epoch=epoch, epochs=epochs)
```
![](16-12-2020/1/1.png)
![](16-12-2020/1/2.png)
![](16-12-2020/1/3.png)
</details>

forest score: 0.943571

## 2

ho fittato una foresta con numeri casuali:
<details>

```
a = np.random.normal(0, 1, (58800,100))

forest2.fit(a, y_train)
```
</details>

score=0.06

# 18/12/2020
cosa succede quando delta = 0?
<details>

```
layer1 = chu.CHUNeuralNetwork()
epochs=160
batch_size=100

start = time()
for epoch in range(epochs):
    X = X_train[np.random.permutation(len(X_train)),:]
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        layer1 = layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0., p=2,
                                         R=1, scale=1, k=2, learn_rate=0.02,
                                         sigma=1, epoch=epoch, epochs=epochs)
    print(epoch)
```

![](18-12-2020/1/1.png)
![](18-12-2020/1/2.png)
![](18-12-2020/1/3.png)
</details>

score = 0.934

# 21/12/2020

## 1

Ho provato con original-modified se non convergeva, e effettivamente non converge! questo è il codice:

<details>

```
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import CHUNeuralNetwork as chu
from time import time
import utilities as utils

np.random.seed(1024)
rng = np.random.default_rng(1024)


# %% loading and splitting the MNIST dataset

if not exists('./database_file'):
    mnist, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    mnist_dataframe = pd.DataFrame(mnist)
    mnist_dataframe.to_hdf("database_file", key="key")

X_train = np.array(pd.read_hdf("database_file"))/255.



# %%
eps0=0.02    # learning rate
Kx=10 # draw parameter
Ky=10 # draw parameter
hiddens=Kx*Ky    # number of hidden units that are displayed in Ky by Kx array
sigma=1.0 # init weight standard deviation
epochs=160      # number of epochs
batch_size=100      # size of the minibatch
prec=1e-30 # safety nonzero division parameter
delta=0.4    # Strength of the anti-hebbian learning
p=2.0        # Lebesgue norm of the weights
k=2          # ranking parameter, must be integer that is bigger or equal than 2

# %%


weight_matrix = np.random.normal(0, sigma, (hiddens, len(X_train[0]))) # init weights
start=time()
for epoch in range(epochs):
    X_train=X_train[np.random.permutation(len(X_train)),:]
    for batch in chu.batchize(X_train, batch_size):
        sig=np.sign(weight_matrix)
        product = batch @ (sig*np.absolute(weight_matrix)**(p-1)).T # (i,j)
        y=np.argsort(product)
        update = np.zeros((hiddens, len(X_train[0])))
        for i in range(len(batch)):
            h = y[i,-1]
            a = y[i,-k]
            update[h] += batch[i] - product[i,h] * weight_matrix[h]
            update[a] += -delta * (batch[i] - product[i,a] * weight_matrix[a])

        scaled_update = chu.scale_update(update, epoch, epochs, eps0)
        weight_matrix += scaled_update
    print(epoch)
print(time()-start)

utils.image_representation(weight_matrix)
```


![](21-12-2020/1/1.png)
![](21-12-2020/1/2.png)
![](21-12-2020/1/3.png)
</details>

# comunicazione
da oggi in poi si lavora con delta = 0

## 2

ho provato con original a mettere il delta = 0 e non converge:
<details>

```

```
</details>
