import numpy as np
import scipy as sp

punkte=np.array[((0,10),(10,0))]

def vec_sub(minuend, subtrahend, list_name):
    result = list_name[minuend] - list_name[subtrahend]
    return(result)

print(vec_sub((1,0),(0,0),punkte))

'''
Vektor-Errechnung:
import numpy as np
v = np.array(entry1) - np.array(entry2)

Sortierung der Liste:
array_sorted = array[np.argsort(array[:, col_index])]
'''
