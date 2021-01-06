import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


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
    return (np.sum(np.abs(matrix) ** p, axis=1) ** (1/p))

def mnist_loader():
    """load up an already nomalized MNIST database.

    Download the MNIST database, thean load it in memory and return it in 4
    numpy arrays.

    return
    ------

    X_train: ndarray
        Shape (45000, 784). 4500 samples FOR EACH FIGURE, ordered and already
        normalized.
    y_train: ndarray
        Shape (45000,). the corresponding labels.
    X_test: ndarray
        Shape (10000, 784). 10000 samples for testing.
    y_test: ndarray
        Shape (10000,). The corresponding labels.
    """
    if not exists('./data/mnist'):
        bunch = fetch_openml('mnist_784', version=1, as_frame=True)
        bunch.frame.to_hdf('data/mnist', key='key', format='table')
    mnist = pd.read_hdf('data/mnist', key='key')
    train, test = train_test_split(mnist, test_size=10000)
    sorted_train = train.sort_values('class')
    equal_train = sorted_train.groupby('class').head(4500).reset_index(drop=True)

    X_train = np.array(equal_train.drop('class', axis=1))/255
    y_train = np.array(equal_train['class'])
    X_test = np.array(test.drop('class', axis=1))/255
    y_test = np.array(test['class'])
    return (X_train, y_train, X_test, y_test)
        


def image_representation(matrix, p, epoch, heatmap, pnorms, ravel):
    """Represent some useful data about the matrix.

    heatmap:
        Represent all synapses in a heatmap image, wich consents to visually
        characterize the shapes emerging while fitting.
    pnorms:
        the p-norms of each row in matrix, wich should converge to the
        parameter R of the CHUNeuralNetwork.
    ravel:
        All the synapses, unraveled.

    Parameters:
    -----------
    matrix
        The matrix of the synapses.
    p
        The exponent of the p-norms.
    heatmap:
        Whenever to show the heatmap
    pnorms:
        Whenever to show the p-norms.
    ravel:
        Whenever to show the unraveled synapses.
    """
    if heatmap:
        image = put_in_shape(matrix, 10, 10, 28, 28)
        vmax = np.amax(np.abs(image))
        im, ax = plt.subplots()
        plt.imshow(image, cmap='bwr', vmax = vmax, vmin=-vmax)
        plt.colorbar()
        plt.title('epochs processed: ' + str(epoch+1))
    if pnorms:
        im2, ax2 = plt.subplots()
        plt.plot(norms(matrix, p))
        plt.title('epochs processed: ' + str(epoch+1))
    if ravel:
        im3, ax3 = plt.subplots()
        plt.plot(np.ravel(matrix))
        plt.title('epochs processed: ' + str(epoch+1))
    
def matrix_saver(matrix, key):
    "save the fit results in a file."
    data = pd.DataFrame(matrix)
    data.to_hdf('results/matrices', key=key)


def score(X_train, y_train, X_test, y_test, transformer, args):
    transformed_train = transformer.transform(X_train, *args)
    transformed_test = transformer.transform(X_test, *args)
    forest1 = RandomForestClassifier()
    forest1.fit(transformed_train, y_train)
    score1 = forest1.score(transformed_test, y_test)
    return score1
    # my score: 0.94
    # no transform: 97
    