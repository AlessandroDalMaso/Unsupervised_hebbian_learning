"""toy model implementation."""
import numpy as np
import CHUNeuralNetwork as chu
import matplotlib.pyplot as plt

np.random.seed(1024)

# %% implementation

data = np.random.normal(0, 223, (50000,784)) #  sqrt(50000)=223
toy = layer1 = chu.CHUNeuralNetwork(n_hiddens=100, scale=1e4, k=3)

toy.fit(data, epochs=5)

# %% image representation

im2, ax2 = plt.subplots()
ax2 = plt.plot(chu.norms(layer1.weight_matrix, 3))

im, ax = plt.subplots()
ax = plt.imshow(toy.weight_matrix)