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
arr = np.zeros([ATP,4], dtype=object)
#the array containing the information about the points, vectors and orthogonal vectors
#arr is sorted [[index of the point (being counted from 1 onwards)],[coordinates of the point],[vector to the next set up point],[normal-vector to the vector to the next point]]
arrT = np.array([[]])
arrT[0] = [1]

arr[1] = np.array([0, [0, 10], [10,-10], [sqrt(0.5),sqrt(0.5)]], dtype=object)
arr[0] = np.array([1, [10, 0], "non-existent", "non-existent"], dtype=object)

#while AIP <= AIP:
    #break
def newpoint(point, vec):
    defpoint = AA(point) + 0.5*AA(vec)
    return defpoint


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
            break
        print(arr)

        #arr_sorted = arr[np.argsort(arr[:,[1]])]
    #return arr_sorted

def normvec(defvector):
    defnormvec = np.array([float(-defvector[0]),float(defvector[1])]) / float(sqrt((defvector[0])**2 + (defvector[1])**2))
    return defnormvec

def vector(minuend, subtrahend):
    result = AA(minuend) - AA(subtrahend)
    return result

sort_arr()
#print(sort_arr())
#print()
#print(vector([0,1],[1,1],arr))

'''
#print(normvec(vector(1,0, punkte)))
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if isinstance(arr[i, j], list):
            arr[i, j] = np.array(arr[i, j])


#print(np.dot([2,5],[35,5]))



#print(vector(arr[0][1],arr[1][1]))

Vektor-Errechnung:
import numpy as np
v = np.array(entry1) - np.array(entry2)

Sortierung der Liste:
array_sorted = array[np.argsort(array[:, col_index])]
'''
end_time = time.process_time()
print(f"Elapsed time: {end_time - start_time} seconds")