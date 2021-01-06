import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
from os.path import exists
import matplotlib.pyplot as plt


def put_in_shape(matrix, rows, columns, height, width, indexes=None):
    """put the synapses in a human-readable shape.

    Each neuron has height x width synapses. These rectangles get represented
    justaxposed in the same image.

    parameters
    ----------

    matrix
        The synapses.
    rows, columns
        The image is composed of rows*columns subimages...
    height, width
        ...each of wich is height*width pixels.
    indexes:
        the indexes of the neurons whose synapses we want to represent. if it
        is None, then all synapses will be represented.

    return
    ------
    image: ndarray
        shape: (rows*width, columns*height)
        
    """
    if indexes is None:
        indexes = range(len(matrix))
    counter = 0
    h = height
    w = width
    image=np.zeros((w*rows, h*columns))
    for y in range(rows):
        for x in range(columns):
            shape = (h, w)
            subimage = np.reshape(matrix[indexes[counter]], shape)
            image[y*h:(y+1)*h, x*w:(x+1)*w] = subimage
            counter += 1
    return image

def norms(matrix, p):
    """p-norms of vectors in a matrix."""
    return np.sum(np.abs(matrix) ** p, axis=1)

def mnist_loader(test_size):
    # only safe for test_size < 0.27!
    if not exists('./data/mnist'):
        bunch = fetch_openml('mnist_784', version=1, as_frame=True)
        bunch.frame.to_hdf('data/mnist', key='key', format='table')
    mnist = pd.read_hdf('data/mnist', key='key')
    train, test = train_test_split(mnist, test_size=test_size)
    sorted_train = train.sort_values('class')
    equal_train = sorted_train.groupby('class').head(4500).reset_index(drop=True)

    X_train = np.array(equal_train.drop('class', axis=1))/255
    y_train = np.array(equal_train['class'])
    X_test = np.array(test.drop('class', axis=1))/255
    y_test = np.array(test['class'])
    return (X_train, y_train, X_test, y_test)
        


def image_representation(matrix, p, heatmap, p_norms, ravel):
    if heatmap:
        image = put_in_shape(matrix, 10, 10, 28, 28)
        vmax = np.amax(np.abs(image))
        im, ax = plt.subplots()
        ax = plt.imshow(image, cmap='bwr', vmax = vmax, vmin=-vmax)
        plt.colorbar()
    if p_norms:
        im2, ax2 = plt.subplots()
        ax2 = plt.plot(norms(matrix, p))
    if ravel:
        im3, ax3 = plt.subplots()
        ax3 = plt.plot(np.ravel(matrix))
    
def matrix_saver(matrix, key):
    data = pd.DataFrame(matrix)
    data.to_hdf('results/matrices', key=key)
    