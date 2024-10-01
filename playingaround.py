import multiprocessing as mp
import math
from math import sqrt
import sys
import numpy as np


'''
sys.set_int_max_str_digits(1000000)

# Funktion für aufwändige Berechnungen
def calc_factorial(number):
    return math.factorial(number)

if __name__ == '__main__':
    # Erstelle eine Liste von Aufgaben
    numbers = [100000, 100000, 100000, 100000, 98273, 29837]

    # Erstelle einen Pool von Prozessen
    pool = mp.Pool(processes=16)  # 4 Prozesse, entsprechend der Anzahl der CPU-Kerne

    # Führe die Aufgaben parallel aus
    results = pool.map(calc_factorial, numbers)

    print(results)



'''
arr = np.array([[1, [5, 1], [2, 1], [0, 0]],
                       [2, [0, 2], [5, 7], [1, 6]],
                       [3, [8, 6], [9, 34], [4, -7.8]],
                       [4, [3, 8], [-2, 8], [2, 56.8]],
                       [5, [1, -6], [0, 1], [3, -6]],
                       [6, [5, -32], [0, 7], [1, 420]],
                       [7, [-3, 32], [234, 3], [69, 2]],
                       [8, [72, 456], [03.5, -69], [2, 3]]], dtype=object)
array = np.array([[0, [0, 10], [10, -10],[0.7071067811865476, 0.7071067811865476]],[1, [10, 0], 'non-existent', 'non-existent'], [0,[0],0,0], [0,[0],0,0]], dtype=object)
'''
ATP = 3
arr = np.zeros([ATP,4,2], dtype=object)
print(arr)
array = [0, [0, 10], [10,-10], [sqrt(0.5),sqrt(0.5)]]
arr[0][0][0] = array[1]
#arr[1][0:] = np.array([1, [10, 0], "non-existent", "non-existent"], dtype=object)
'''

print(arr[:6])

'''
array_sorted = arr[np.argsort(arr[:,[1][0]])]
print('\n'*7, arr[0],"\n"*3,  array)
'''


