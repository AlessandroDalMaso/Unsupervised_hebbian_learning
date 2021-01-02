import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from utilities import image_representation

X = np.array(pd.read_hdf('matrices', key='random'))

r_link = linkage(X, metric='cosine', method='average')

indexes = fcluster(r_link, t=0.3, criterion='distance')
print(np.amax(indexes))

for i in range(15):
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
