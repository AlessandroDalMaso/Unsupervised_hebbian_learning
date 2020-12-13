"""Instance CHUNeuralNetwork, fit, transform, represent weights as images."""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import CHUNeuralNetwork as chu
from time import time

np.random.seed(1024)


# %% loading, splitting  and normalizing the MNIST dataset

if not exists('./data/mnist'):
    bunch = fetch_openml('mnist_784', version=1, as_frame=True)
    bunch.frame.to_hdf('data/mnist', key='key', format='table')
database = pd.read_hdf('data/mnist', key='key')