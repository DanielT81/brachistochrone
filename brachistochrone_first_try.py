#where to continue: sort_arr()

import numpy as np
from math import sin, cos, pi, sqrt
from numpy import asarray as AA
import scipy as sp
import time
start_time = time.process_time() #to track the computation-time
for i in range(1000000):
    pass

index_number = 2
AIP = 4 #amount of iterations to create new points
ATP = 2**AIP + 1 #amount of total points in the system
arr_len = 2 #length of the non-zero values
arr = np.zeros([ATP,2], dtype=object) #array with all the points and given indices to track manually
arrT = np.zeros([2,8], dtype=object) #read the README to get the structure

arr[0] = np.array([0, [0, 10], dtype=object) #setting the boundary conditions
arr[1] = np.array([1, [10, 0], dtype=object) #setting the boundary conditions
arrT[0] = [[0,1], arr[0,1], arr[1,1], new_point(arr[0,1], vector(arr[0,1],arr[1,1])), norm_vec(vector(arr[0,1],arr[1,1])), 0, ] #setting the boundary condition so there is no need for an annoying if clause
def new_point(point, vec):
    defpoint = AA(point) + 0.5*AA(vec)
    return defpoint


    #
def sort_arr():
    iterations = 0
    total_iterations = np.size(new_arr, 0)
    while iterations <= total_iterations:
        if new_arr[iterations, 1] != 0.0:
            iterations +=1
        else:
            new_arr[:iterations] = new_arr[np.argsort(new_arr[:iterations, [1][0]])]
            arr_len = iterations
            break


    #
def norm_vec(defvector):
    defnormvec = np.array([float(-defvector[0]),float(defvector[1])]) / float(sqrt((defvector[0])**2 + (defvector[1])**2))
    return defnormvec


    #
def vector(minuend, subtrahend):
    result = AA(minuend) - AA(subtrahend)
    return result


    #

def vec_arr():
    iterations = 0


#print(sort_arr())
#print()
#print(vector([0,1],[1,1],arr))

'''
#print(normvec(vector(1,0, punkte)))
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if isinstance(arr[i, j], list):
            arr[i, j] = np.array(arr[i, j])

'''
end_time = time.process_time()
print(f"Elapsed time: {end_time - start_time} seconds")