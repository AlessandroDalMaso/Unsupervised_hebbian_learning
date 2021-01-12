import numpy as np
from numpy.linalg import norm

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from utilities import image_representation
from math import sqrt
from hausdorff import hausdorff_distance
from scipy.spatial.distance import squareform

def dist_cos(v, u):
        return 1 - (v @ u.T) / (norm(u) * norm(v))

def three_d(v):
    points = np.empty((len(v),3))
    lenght = int(sqrt(len(v)))
    img = np.reshape(v, (lenght, lenght))
    valmax = np.amax(img)
    for (y, x), val in np.ndenumerate(img):
        points[lenght*y+x] = np.array([x/28, y/28, val])
    return points

def dist_haus(u, v, pdist='euclidean'):
        u3 = u.reshape(len(u)//3, 3)
        v3 = v.reshape(len(v)//3, 3)
        return hausdorff_distance(u3, v3, pdist)

def dist_row(u, v):
        u_sq = u.reshape((28,28))
        v_sq = v.reshape((28,28))
        u_r = u_sq.sum(axis=1)
        v_r = v_sq.sum(axis=1)
        return dist_cos(u_r, v_r)

def littleroot(Aw_ij, Bw_ij, A_ij, B_ij):
    d1 = hausdorff_distance(Aw_ij, np.array([B_ij]))
    d2 = hausdorff_distance(Bw_ij, np.array([A_ij]))
    return sqrt(d1 * d1 + d2 * d2)

def dist_avg(u, v):
    A = u[:,:,2].reshape((28,28))
    B = v[:,:,2].reshape((28,28))
    W = 5
    N = len(A)
    total = 0
    for ay in np.arange(W, N-W, 1):
        for ax in np.arange(W, N-W, 1):
            for by in np.arange(W, N-W, 1):
                    for bx in np.arange(W, N-W, 1):
                        Aw_ij = A[ay-W:ay+W, ax-W,ay+W]
                        Bw_ij = B[by-W:by+W, bx-W,by+W]
                        A_ij = A[]
                        total += littleroot(Aw_ij, Bw_ij, A_ij, B_ij)
                
        

random = np.array(pd.read_hdf('results/matrices', key='random'))
mono = np.array(pd.read_hdf('results/matrices', key='monotype'))
_1v1 = np.array(pd.read_hdf('results/matrices', key='_1v1'))

random_3d = np.apply_along_axis(three_d, 1, random).reshape((100,784*3))
mono_3d = np.apply_along_axis(three_d, 1, mono).reshape((100,784*3))
_1v1_3d = np.apply_along_axis(three_d, 1, _1v1).reshape((100,784*3))


r_link = linkage(random_3d, metric=dist_haus, method='average')

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
for i in np.arange(0.08, 0.12, 0.001):
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



