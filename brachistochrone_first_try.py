import numpy as np
from math import sqrt
#import scipy as sp

#punkte = np.array(((0,10),(10,0)))


punkte=np.array([[[0,10],[10,-10]],[[-sqrt(0.5),0],[-sqrt(0.5),0]] , [[10,0],[0,0]]])


def normvec(defvector):
    defother = defvector / np.linalg.norm(defvector)
    defnormvec = defvector - np.dot(defvector, defother)[:, None] * defother
    return defnormvec

def vector(minuend, subtrahend, list_name):
    result = list_name[minuend] - list_name[subtrahend]
    return(result)


print(normvec(vector(1,0, punkte)))
print(punkte)


#print(vector((1,1),(0,1),punkte))
'''
Vektor-Errechnung:
import numpy as np
v = np.array(entry1) - np.array(entry2)

Sortierung der Liste:
array_sorted = array[np.argsort(array[:, col_index])]
'''
