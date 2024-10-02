import numpy as np
from numba.np.arrayobj import array_dtype
from numpy import asarray as AA
from math import sqrt

arr = np.array([[1, [5, 1], [2, 1], [0, 0]],
                       [2, [0, 2], [5, 7], [1, 6]],
                       [3, [8, 6], [9, 34], [4, -7.8]],
                       [4, [3, 8], [-2, 8], [2, 56.8]],
                       [5, [1, -6], [0, 1], [3, -6]],
                       [6, [5, -32], [0, 7], [1, 420]],
                       [7, [-3, 32], [234, 3], [69, 2]],
                       [8, [72, 456], [03.5, -69], [2, 3]],
                       [0, 0, 0, 0]], dtype=object)


array = np.array([[[5, 1], [2, 1], [0, 0]],
                       [[0, 2], [5, 7], [1, 6]],
                       [[8, 6], [9, 34], [4, -7.8]],
                       [[3, 8], [-2, 8], [2, 56.8]],
                       [[1, -6], [0, 1], [3, -6]],
                       [[5, -32], [0, 7], [1, 420]],
                       [[-3, 32], [234, 3], [69, 2]],
                       [[72, 456], [03.5, -69], [2, 3]]])
new_arr = np.array([])
def sort_arr():
    iterations = 0
    total_iterations = np.size(arr, 0)
    while iterations <= total_iterations:
        if arr[iterations][1] != 0:
            iterations+=1
        else:
            #print(arr[:iterations], '\n'*10)
            arr[:iterations] = arr[np.argsort(arr[:iterations,[1][0]])]
            #arr = np.add(arr[np.argsort(arr[:iterations, [1][0]])],np.zeros([ATP-iterations, 4], dtype=object))
            #print(arr[np.argsort(arr[:iterations,[1][0]])])
            new_arr = arr
            print(5)
        print(4)
        break

def sort_arr2():
    iterations = 0
    total_iterations = np.size(arr, 0)
    while iterations <= total_iterations:
        if arr[iterations][1] == 0:
            arr[:iterations] = arr[np.argsort(arr[:iterations, [1][0]])]
            print(arr[np.argsort(arr[:iterations, [1][0]])], '\n'*10, 'finally')
            break
        else:
            iterations +=1
            print(iterations)
        print(4)
        break
        '''
        if arr[8][1] != 0:
            new_arr = arr[np.argsort(arr[:iterations, [1][0]])]
        return new_arr
        '''
    #print(arr)
sort_arr2()
print(sort_arr2())
