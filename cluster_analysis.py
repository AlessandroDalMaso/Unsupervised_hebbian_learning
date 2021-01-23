import numpy as np
from numpy.linalg import norm

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from utilities import image_representation
from math import sqrt
from utilities import dist_haus, three_d

def dist_cos(v, u):
        return 1 - (v @ u.T) / (norm(u) * norm(v))


def dist_row(u, v):
        u_sq = u.reshape((28,28))
        v_sq = v.reshape((28,28))
        u_r = u_sq.sum(axis=1)
        v_r = v_sq.sum(axis=1)
        return dist_cos(u_r, v_r)

def littleroot(Aw_ij, Bw_ij, A_ij, B_ij):
    d1 = np.amin((Aw_ij - B_ij) ** 2)
    d2 = np.amin((Bw_ij - A_ij) ** 2)
    return sqrt(d1 * d1 + d2 * d2)

def dist_avg(u, v, W):
    A = u.reshape((28,28))
    B = v.reshape((28,28))
    N = len(A)
    result = 0
    for i in np.arange(W, N-W, 1):
        for j in np.arange(W, N-W, 1):
            Aw_ij = three_d(A[i-W:i+W+1, j-W:j+W+1].ravel())
            Bw_ij = three_d(B[i-W:i+W+1, j-W:j+W+1].ravel())
            A_ij = np.array([i/28, j/28, A[i, j]])
            B_ij = np.array([i/28, j/28, B[i, j]])

            result += littleroot(Aw_ij, Bw_ij, A_ij, B_ij)
            coefficient = 0.5 * (N - 2* W)
    return result * coefficient


random = np.array(pd.read_hdf('results/matrices', key='random'))
mono = np.array(pd.read_hdf('results/matrices', key='monotype'))
_1v1 = np.array(pd.read_hdf('results/matrices', key='_1v1'))



# %% linkage

r_link = linkage(random, metric=dist_haus, method='average')

def g(r_link, dist):
    indexes = fcluster(r_link, t=dist, criterion='distance')
    return (np.amax(indexes))


im, ax = plt.subplots()
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(
    r_link,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

d = []
n = []
for i in np.arange(0, 1, 0.01):
    print(i, g(r_link, i))
    d.append(i)
    n.append(g(r_link, i))
im, ax = plt.subplots()
plt.scatter(d, n)
plt.grid()

# %% fcluster

indexes = fcluster(r_link, t=0.095, criterion='distance')
print(np.amax(indexes))

def represent_cluster(index):
    M = np.zeros((280,280))
    for y in range(10):
        for x in range(10):
            if index == indexes[10*y+x]:
                M[y*28:(y+1)*28,x*28:(x+1)*28] = random[10*y+x].reshape((28,28))
    im, ax = plt.subplots()
    vmax = np.amax(np.abs(M))
    plt.imshow(M, cmap='bwr', vmax=0.2, vmin=-0.2)
    plt.colorbar()
    plt.title('cluster '+str(index))

for i in range(0, np.amax(indexes)+1):
    represent_cluster(i)

image_representation(random, 2, 159, True, False, False)



