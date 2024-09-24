import numpy as np
from math import sin, cos, pi, sqrt
import scipy as sp
import time
start_time = time.process_time() #to track the computation-time
for i in range(1000000):
    pass


AIP = 20 #amount of iterations to create new points
ATP = 2**AIP + 1 #amount of total points in the system
arr = np.zeros([ATP,4], dtype=object) #the array containing the information about the points, vectors and orthogonal vectors
#arr is sorted [[index of the point (being counted from 1 onwards)],[coordinates of the point],[vector to the next set up point],[normal-vector to the vector to the next point]]

arr[0] = np.array([0, [0, 10], [10,-10], [sqrt(0.5),sqrt(0.5)]], dtype=object)
arr[ATP-1] = np.array([ATP, [10, 0], "non-existent", "non-existent"], dtype=object)



def normvec(defvector):
    defother = defvector / np.linalg.norm(defvector)
    defnormvec = defvector * np.dot(defvector, defother)[:, None] * defother
    return defnormvec

def vector(minuend, subtrahend, list_name):
    result = list_name[minuend] - list_name[subtrahend]
    return(result)

'''
#print(normvec(vector(1,0, punkte)))
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if isinstance(arr[i, j], list):
            arr[i, j] = np.array(arr[i, j])
'''

print(np.dot([2,5],[35,5]))
end_time = time.process_time()
print(f"Elapsed time: {end_time - start_time} seconds")


#print(vector((1,1),(0,1),punkte))
'''
Vektor-Errechnung:
import numpy as np
v = np.array(entry1) - np.array(entry2)

Sortierung der Liste:
array_sorted = array[np.argsort(array[:, col_index])]
'''
