import numpy as np

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