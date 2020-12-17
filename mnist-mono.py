import numpy as np
import pandas as pd
np.random.seed(1024)

mnist = pd.read_hdf('data/mnist', key='key')
groups = mnist.groupby('class')
groups.groups
a = list(groups.__iter__())
