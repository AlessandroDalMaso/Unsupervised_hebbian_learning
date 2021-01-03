import numpy as np
from numpy.linalg import norm

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import squareform
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from utilities import image_representation

X = np.array(pd.read_hdf('matrices', key='random'))

def dist(v, u):
        return 1 - (v @ u.T) / (norm(u) * norm(v))


r_link = linkage(X, metric=dist, method='average')

indexes = fcluster(r_link, t=0.33, criterion='distance')
print(np.amax(indexes))

for i in range(np.amax(indexes)+1):
    M = np.zeros((280,280))
    for y in range(10):
        for x in range(10):
            if i == indexes[10*y+x]:
                M[y*28:y*28+28,x*28:x*28+28] = X[10*y+x].reshape((28,28))
    im, ax = plt.subplots()
    vmax = np.amax(np.abs(M))
    plt.imshow(M, cmap='bwr', vmax=0.2, vmin=-0.2)
    plt.colorbar()

image_representation(X, 2, True, False, False)

def g(r_link, dist):
    indexes = fcluster(r_link, t=dist, criterion='distance')
    return (np.amax(indexes))

d = []
n = []
for i in np.arange(0, 1, 0.01):
    print(i, g(r_link, i))
    d.append(i)
    n.append(g(r_link, i))
    
plt.plot(d, n)
