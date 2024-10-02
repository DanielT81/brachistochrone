import numpy as np
from numba.np.arrayobj import array_dtype
from numpy import asarray as AA
from math import sqrt
import time
start_time = time.process_time() #to track the computation-time
for i in range(1000000):
    pass

arr = np.array([[1, [5, 1], [2, 1], [0, 0]],
                       [2, [-1, 2], [5, 7], [1, 6]],
                       [3, [8, 6], [9, 34], [4, -7.8]],
                       [4, [3, 8], [-2, 8], [2, 56.8]],
                       [5, [1, -6], [0, 1], [3, -6]],
                       [6, [5, -32], [0, 7], [1, 420]],
                       [7, [-3, 32], [234, 3], [69, 2]],
                       [8, [72, 456], [03.5, -69], [2, 3]],
                       [0, 0, 0, 0]], dtype=object)


rng = np.random.default_rng()
new_arr = np.append(rng.uniform(low=0.01, high=10, size=[10000000,4]), np.zeros([400,4]), axis=0)


def sort_arr():
    iterations = 0
    total_iterations = np.size(new_arr, 0)
    while iterations <= total_iterations:
        if new_arr[iterations, 1] != 0.0:
            iterations +=1
        else:
            new_arr[:iterations] = new_arr[np.argsort(new_arr[:iterations, [1][0]])]
            break


def sort_arr2():
    iterations = 0
    total_iterations = np.size(new_arr, 0)
    step = int(round(total_iterations/10_000, 0)) + 1
    while True:
        try:
            while new_arr[iterations, 1] != 0:
                iterations += step
            iterations -= step
            while 1==1:
                if new_arr[iterations, 1] != 0.0:
                    iterations += 1
                else:
                    print('got it')
                    break
        except IndexError:
            iterations -=1
            while 1 == 1:
                if new_arr[iterations, 1] != 0.0:
                    iterations += 1
                else:
                    print('got it')
                    break
        finally:
            new_arr[:iterations] = new_arr[np.argsort(new_arr[:iterations, [1][0]])]
            break

sort_arr2()
print(new_arr[:100])



end_time = time.process_time()
print(f"Elapsed time: {end_time - start_time} seconds")