import numpy as np
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

def normvec(defvector):
    defnormvec = np.array([float(-defvector[0]),float(defvector[1])]) / float(sqrt((defvector[0])**2 + (defvector[1])**2))
    return defnormvec




#array_sorted = array[np.argsort(array[:, [1][0]])]
print(normvec(array[2][1]))
