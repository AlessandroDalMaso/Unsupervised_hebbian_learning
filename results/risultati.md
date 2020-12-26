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

Ho provato, con delta = 0.4, a non scramblare la batch internamente:

<details>

```
"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np

import CHUNeuralNetwork as chu
from time import time
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import utilities as utils
import random
np.random.seed(1024)
random.seed(0)

(X_train, y_train, X_test, y_test) = utils.mnist_loader(test_size=0.16)
batch_size=100


# %% fit the data

layer1 = chu.CHUNeuralNetwork()
epochs=160
X = X_train[np.random.permutation(len(X_train))]
batches=[]
for i in range(0, len(X_train), batch_size):
    batches.append(X[i:i+batch_size])

start = time()
for epoch in range(epochs):
    shuffle(batches)
    for batch in batches:
        layer1 = layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0, p=2,
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
# no transform: 97
```

![](21-12-2020/2/1.png)
![](21-12-2020/2/2.png)
![](21-12-2020/2/3.png)
</details>

## 3

Ho provato anche questa strana versione intermedia:

<details>

```
"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np

import CHUNeuralNetwork as chu
from time import time
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import utilities as utils
import random
np.random.seed(1024)
random.seed(0)

(X_train, y_train, X_test, y_test) = utils.mnist_loader(test_size=0.16)
batch_size=100


# %% fit the data

layer1 = chu.CHUNeuralNetwork()
epochs=160


start = time()
for epoch in range(epochs):
    batches=[]
    X = X_train[np.random.permutation(len(X_train))]
    for i in range(0, len(X_train), batch_size):
        batches.append(X[i:i+batch_size])
    X = X_train[np.random.permutation(len(X_train))]
    for batch in batches:
        layer1 = layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0, p=2,
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
# no transform: 97
```

![](21-12-2020/3/1.png)
![](21-12-2020/3/2.png)
![](21-12-2020/3/3.png)

</details>

# 22/12/2020

## 1

Ho provato per controllare se il random seed funziona. funziona.

<details>

```
"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np

import CHUNeuralNetwork as chu
from time import time
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import utilities as utils
import random
np.random.seed(1024)
random.seed(0)

(X_train, y_train, X_test, y_test) = utils.mnist_loader(test_size=0.16)
batch_size=100


# %% fit the data

layer1 = chu.CHUNeuralNetwork()
epochs=10


start = time()
for epoch in range(epochs):
    batches=[]
    X = X_train[np.random.permutation(len(X_train))]
    for i in range(0, len(X_train), batch_size):
        batches.append(X[i:i+batch_size])
    X = X_train[np.random.permutation(len(X_train))]
    for batch in batches:
        layer1 = layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0, p=2,
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
# no transform: 97
```

![](22-12-2020/1/1.png)
![](22-12-2020/1/1.png)
</details>

## 2

è successa la cosa più strana... ho semplicemnte semplificato la procedura di creazione delle batch e adesso converge...

<details>

questo è il codice che converge:

```
"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np

import CHUNeuralNetwork as chu
from time import time
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import utilities as utils
import random
np.random.seed(1024)
random.seed(0)

(X_train, y_train, X_test, y_test) = utils.mnist_loader(test_size=0.16)
batch_size=100


# %% fit the data

layer1 = chu.CHUNeuralNetwork()
epochs=60

start = time()
for epoch in range(epochs):
    X = X_train[np.random.permutation(len(X_train))]
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        layer1 = layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0.4
                                         , p=2,
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
# no transform: 97

```

questo è il ciclo for che non converge:

```
for epoch in range(epochs):
    batches=[]
    X = X_train[np.random.permutation(len(X_train))]
    for i in range(0, len(X_train), batch_size):
        batches.append(X[i:i+batch_size])
    X = X_train[np.random.permutation(len(X_train))]
    for batch in batches:
        layer1 = layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0, p=2,
                                         R=1, scale=1, k=2, learn_rate=0.02,
                                         sigma=1, epoch=epoch, epochs=epochs)
    print(epoch)
```

ecco i risultati del codice che converge:

![](22-12-2020/2/1.png)
![](22-12-2020/2/2.png)
![](22-12-2020/2/3.png)


</details>

## 3

Ho provato a mettere:
* una funzione triangolare per inizializzare la matrice, con sigma come left and right;
* sigma = 0.01;
* average invece di amax in scale_update.

risultato: molto brutto. score 9%

<details>

![](22-12-2020/3/1.png)
![](22-12-2020/3/2.png)
![](22-12-2020/3/3.png)

</details>

## 4

Ho fatto varie prove variando la sigma e la distribuzione di probabilità.

Innanzitutto: a 160 batch con sigma=1, la gaussiana non diverge, mentre la distribuzione triangolare sì, ecco cosa ottengo con la triangolare:

<details>

![](22-12-2020/4/tri/160/sigma_is_1/1.png)
![](22-12-2020/4/tri/160/sigma_is_1/2.png)
![](22-12-2020/4/tri/160/sigma_is_1/3.png)

</details>

Poi: ho provato a diminuire la sigma a 0.1: e niente, vengono sempre i 9 sia con i triangoli che con le gaussiane! (molto interessante)

<details>

![](22-12-2020/4/nein/1.png)
![](22-12-2020/4/nein/2.png)
![](22-12-2020/4/nein/3.png)

</details>


Faccio infine un altro backup del codice che mi da le cifre perché sono paranoico:


<details>

```
"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np

import CHUNeuralNetwork as chu
from time import time
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import utilities as utils
import random
np.random.seed(1024)
random.seed(0)

(X_train, y_train, X_test, y_test) = utils.mnist_loader(test_size=0.16)
batch_size=100


# %% fit the data

layer1 = chu.CHUNeuralNetwork()
epochs=160




start = time()

for epoch in range(epochs):
    X = X_train[np.random.permutation(len(X_train))]
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        layer1 = layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0.4
                                         , p=2,
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
# no transform: 97
```

</details>

## 5

Ho provato a mettere k=7. (sigma=1, 60 epochs)

<details>

</details>

# 23/12/2020

## 1

Ecco DI NUOVO i RISULTATI che CONVERGONO:

(score=94%)

<details>

![](23-12-2020/1/1.png)
![](23-12-2020/1/2.png)
![](23-12-2020/1/3.png)


```
"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np

import CHUNeuralNetwork as chu
from time import time
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import utilities as utils
import random
np.random.seed(1024)
random.seed(0)

(X_train, y_train, X_test, y_test) = utils.mnist_loader(test_size=0.16)
batch_size=100


# %% fit the data

layer1 = chu.CHUNeuralNetwork()
epochs=160


start = time()

for epoch in range(epochs):
    X = X_train[np.random.permutation(len(X_train))]
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        layer1 = layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0.4
                                         , p=2,
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
# no transform: 97
```

</details>

# 24/12/2020

## 1

Ho fatto un po' di prove sul perché non converge quando le p-norme sono tutte 1. succede che uno a caso dei vettori viene selezionato più di tutti. questi sono gli indici del vettore massimo ogni volta:

[11, 11,  5, 48, 67, 11, 67, 11, 11, 53, 11, 11, 67, 79,  4,  5, 30,
       11, 76, 74, 78, 11, 11, 74, 11, 11, 11,  5, 30, 53, 40, 92, 78, 48,
       92, 74, 11, 64,  8, 35, 63, 78, 78, 11, 27, 92, 11, 30, 78, 51, 76,
       11, 11, 66,  4, 79, 15,  9, 78, 30,  5, 78, 11, 54, 66, 30, 67,  4,
       90, 78, 66, 11, 58, 77, 11, 11, 48, 78, 30, 11, 54, 11, 11, 37, 58,
       11, 73, 78, 11, 66, 86, 78, 11, 40, 10, 30, 54, 11, 11, 64]

ed ecco cosa succede alle p-norm dopo una batch:

<details>

![](24-12-2020/1/1.png)

</details>

dopo due batch, la situazione è peggiorata:

<details>

![](24-12-2020/1/2.png)

</details>

## finalmente, convergenza, con sigma=10:

<details>

```
"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
import CHUNeuralNetwork as chu
from time import time
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import utilities as utils
import random
np.random.seed(1024)
random.seed(0)

(X_train, y_train, X_test, y_test) = utils.mnist_loader(test_size=10000)
batch_size=100


# %% fit the data

layer1 = chu.CHUNeuralNetwork()
epochs=160


start = time()

for epoch in range(epochs):
    X = X_train[np.random.permutation(len(X_train))]
    batches = X.reshape((45000//batch_size, batch_size, 784))
    for batch in batches:
        layer1 = layer1.fit_single_batch(batch=batch, n_hiddens=100, delta=0.4,
                                         p=2,
                                         R=1, scale=1, k=2, learn_rate=0.02,
                                         sigma=10, epoch=epoch, epochs=epochs)
    print(epoch)
print(time()-start)

utils.image_representation(layer1.weight_matrix, 2)


# %% second layer


transformed_train = layer1.transform(X_train, chu.activ, 4.5)
transformed_test = layer1.transform(X_test, chu.activ, 4.5)

forest1 = RandomForestClassifier()

start=time()
forest1.fit(transformed_train, y_train)
print(time()-start)

score1 = forest1.score(transformed_test, y_test)
# my score: 0.94
# no transform: 97
```

![](24-12-2020/2/1.png)
![](24-12-2020/2/2.png)
![](24-12-2020/2/3.png)

</details>

# Natale

## 1

Un po' di esplorazione dello spazio delle fasi oggi. Selezionati 5 parametri promettenti, prima di tutto prendo dei punti a caso nel loro spazio delle fasi, e li vado a confrontare con il valore che so convergere:

<details>

&Delta; = [0, 0.2, 0.4, 0.6, 0.8]

p = [2, 3, 4.5]

k= [2, 3, 4, 5, 6, 7]

&sigma; = [6, 8, 10, ]

LR = [0.01, 0.02, 0.03]

|&Delta;|p|k| &sigma;|LR|score|
|-|-|-|-|-|-|
|0.4|2  |2|10|0.02|94.01|
|0.8|3  |5|6 |0.01|28|
|0  |4.5|3|10|0.02|92|

quindi decido di partire dal primo. Cambio di relativamente poco tre parametri scelti a caso:

|&Delta;|p|k| &sigma;|LR|score|
|-|-|-|-|-|-|
|0.5|3  |3|10|0.02|94.71|

Ok, sta chiaramente succedendo qualcosa di interessante. Visivamente si vedono delle zone negative, però non c'è convergenza. forse gli serve più tempo per convergere? comunque l'opzione è scartata. torno al modelo di base, quello della prima riga della tabella.

provo questa combinazione:

|&Delta;|p|k| &sigma;|LR|score|
|-|-|-|-|-|-|
|0.4|2  |3|10|0.02|94.49|

non converge. La scarto e torno al modello di base. Stavolta cambio solo una variabile:

|&Delta;|p|k| &sigma;|LR|score|
|-|-|-|-|-|-|
|0.4|3  |2|10|0.02|94.28|

 L'ultima prova:

 |&Delta;|p|k| &sigma;|LR|score|
 |-|-|-|-|-|-|
 |0.4|1  |2|10|0.02||

 Niente, non converge.

 </details>

 Poi ho provato con vari valori di n:

<details>

 |n|1|3|4.5|5|
 |-|-|-|-|-|
 |score|93.92|93.99|94.01|93.99|

 </details>

 Poi ho provato a varire le dimensioni delle bach:

 Ora mi manca di provarlo con i tre splitting fatti:

<details>

|batch_size|score|
|-|-|
|200|neuroni che non convergono|
|100|94.01|
|90|93.97|
|50|93.54|

</details>

## 2

Ho provato con le tre suddivisioni in batch:

<details>

### batch Random

score = 94.01%

![](24-12-2020/2/1.png)
![](24-12-2020/2/2.png)
![](24-12-2020/2/3.png)


### batch monotematiche

score=94.34

(no convergenza)

![](26-12-2020/1/1.png)
![](26-12-2020/1/2.png)
![](26-12-2020/1/3.png)


### batch 1v1

score = 93.75

(no convergenza)

![](26-12-2020/2/1.png)
![](26-12-2020/2/2.png)
![](26-12-2020/2/3.png)

</details>
