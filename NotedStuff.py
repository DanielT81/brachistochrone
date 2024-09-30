import numpy as np

array = np.array([[1, [5, 1], [2, 1], [3, 4]],
                          [2, [0, 2], [5, 7], [1, 1]],
                          [3, [8, 6], [9, 3], [4, 2]],
                          [4, [3, 9], [7, 8], [2, 3]]], dtype=object)

arr = np.array([[1, [5, 1], [2, 1], [0, 0]],
                       [2, [0, 2], [5, 7], [1, 6]],
                       [3, [8, 6], [9, 34], [4, -7.8]],
                       [4, [3, 8], [-2, 8], [2, 56.8]],
                       [5, [1, -6], [0, 1], [3, -6]],
                       [6, [5, -32], [0, 7], [1, 420]],
                       [7, [-3, 32], [234, 3], [69, 2]],
                       [8, [72, 456], [03.5, -69], [2, 3]]], dtype=object)

#to access a coordinate in the array
#for i in iter(arr):
    #print(i[0]) #index of the array
    #print(i[1][0]) #first entry of the second part if the array
#def sort():
    #array_sorted = array[np.argsort(array[:,[1][0]])]