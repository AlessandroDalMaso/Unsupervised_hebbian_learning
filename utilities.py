import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
from os.path import exists

def batchize(iterable, batch_size):
    """Put iterables in batches."""
    for n in range(0, len(iterable), batch_size):
        yield iterable[n:min(n + batch_size, len(iterable))]

def put_in_shape(matrix, rows, columns, indexes=None):
    """represent some weights"""
    if indexes is None:
        indexes = range(len(matrix))
    counter = 0
    image=np.zeros((28*rows, 28*columns))
    for y in range(rows):
        for x in range(columns):
            shape = (28, 28)
            subimage = np.reshape(matrix[indexes[counter]], shape)
            image[y*28:(y+1)*28, x*28:(x+1)*28] = subimage
            counter += 1
    return image

def norms(matrix, p):
    """p-norms of vectors in a matrix."""
    return np.sum(np.abs(matrix) ** p, axis=1)

def mnist_loader():
    if not exists('./data/mnist'):
        bunch = fetch_openml('mnist_784', version=1, as_frame=True)
        bunch.frame.to_hdf('data/mnist', key='key', format='table')
    mnist = pd.read_hdf('data/mnist', key='key')
    train, test = train_test_split(mnist, test_size=0.25)
    
    X_train = np.array(train.drop('class', axis=1))/255
    y_train = np.array(train['class'])
    X_test = np.array(test.drop('class', axis=1))/255
    y_test = np.array(test['class'])
    return (X_train, y_train, X_test, y_test)