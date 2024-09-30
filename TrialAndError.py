import numpy as np
from numba.np.arrayobj import array_dtype
from numpy import asarray as AA
from math import sqrt

array = np.array([[1, [5, 1], [2, 1], [0, 0]],
                       [2, [0, 2], [5, 7], [1, 6]],
                       [3, [8, 6], [9, 34], [4, -7.8]],
                       [4, [3, 8], [-2, 8], [2, 56.8]],
                       [5, [1, -6], [0, 1], [3, -6]],
                       [6, [5, -32], [0, 7], [1, 420]],
                       [7, [-3, 32], [234, 3], [69, 2]],
                       [8, [72, 456], [03.5, -69], [2, 3]]], dtype=object)

arr = np.array([[[5, 1], [2, 1], [0, 0]],
                       [[0, 2], [5, 7], [1, 6]],
                       [[8, 6], [9, 34], [4, -7.8]],
                       [[3, 8], [-2, 8], [2, 56.8]],
                       [[1, -6], [0, 1], [3, -6]],
                       [[5, -32], [0, 7], [1, 420]],
                       [[-3, 32], [234, 3], [69, 2]],
                       [[72, 456], [03.5, -69], [2, 3]]])

ind = np.sort(array[:,1:],0)
print(ind)

ind = ind.astype(object)# sorts along first axis (down)
q = np.append([ind,array[:,:1]])#, "\n"*5, array[:,0:1]
q = np.reshape(q, [4,8])
inverted = np.flip(q, 1)
#print(np.shape(q), "\n"*5, q[0])
print(q)
#np.take_along_axis(x, ind, axis=0)  # same as np.sort(x, axis=0)
