import numpy as np
from numpy.linalg import norm

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from utilities import image_representation
from math import sqrt
from utilities import dist_haus, three_d
from PIL import Image

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

num=0

def dist_avg(u, v, W=5):
    A = u.reshape((28,28))
    B = v.reshape((28,28))
    N = len(A)
    result = 0
    for i in np.arange(W, N-W, 1):
        global num
        num+=1
        print(num)
        for j in np.arange(W, N-W, 1):
            Aw_ij = three_d(A[i-W:i+W+1, j-W:j+W+1].ravel())
            #TODO: divide x, y by 28!!!
            Bw_ij = three_d(B[i-W:i+W+1, j-W:j+W+1].ravel())
            A_ij = np.array([i/28, j/28, A[i, j]])
            B_ij = np.array([i/28, j/28, B[i, j]])

            result += littleroot(Aw_ij, Bw_ij, A_ij, B_ij)
            coefficient = 0.5 * (N - 2* W)
    return result * coefficient


def GR(a_ij, b_lm):
    return abs(a_ij - b_lm) / max( a_ij, b_lm, 1e-29 )

def cityblock (i,j,l,m):
    return max(abs(i-l),abs(j-m))

def sigma(A_ij, B_lm, alpha, beta, N):
    (i, j, a_ij) = A_ij
    (l, m, b_lm) = B_lm
    addend_1 = alpha * cityblock(i,j,l,m) / (2*N)
    addend_2 = beta * GR(a_ij, b_lm)
    return addend_1 + addend_2

def GD(A, B, N=28, alpha=0.5, beta=0.5):
    # TODO: understand why GD(x,x) is zero
    A3 = three_d(A)
    B3 = three_d(B)
    coeff = 2/(N*N*(N*N-1))
    total = 0
    for A_ij in A3:
        for B_lm in B3:
            total += sigma(A_ij, B_lm, alpha, beta, N)
    return coeff*total

def rotate(image, angle):
    im = np.array(Image.fromarray(random[1].reshape((28,28))).rotate(angle))
    return im.ravel()

def dist_gyro(A, B):
    sov = 0
    angle_f = 0
    for angle in range(45):
        sov2 = max(A @ rotate(B, angle).T, B @ rotate(A, angle).T)
        if sov2 > sov:
            sov = sov2
            angle_f = angle
        print(angle_f)
    diff1 = A - rotate(B, angle_f)
    diff2 = B - rotate(A, angle_f)
    return max(diff1@diff1.T, diff2@diff2.T)

random = np.array(pd.read_hdf('results/matrices', key='nodelta0'))

# %% linkage

r_link = linkage(random, metric=dist_haus, method='average')

    
# best one: dist-haus

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
    p=18,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

d = []
n = []
for i in np.arange(0.075, 0.1, 0.001):
    print(i, g(r_link, i))
    d.append(i)
    n.append(g(r_link, i))
im, ax = plt.subplots()
plt.scatter(d, n)
plt.grid()

# %% fcluster

indexes = fcluster(r_link, t=0.09, criterion='distance')
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



