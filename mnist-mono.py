import numpy as np
import pandas as pd
import utilities as utils
import CHUNeuralNetwork as chu
from time import time
from sklearn.ensemble import RandomForestClassifier
np.random.seed(1024)

(X_train, y_train, X_test, y_test) = utils.mnist_loader(test_size=0.16)
# loads 5000 samples for each figure, ordered in the array.

layer1 = chu.CHUNeuralNetwork()
epochs_per_figure=1
batch_size=99
