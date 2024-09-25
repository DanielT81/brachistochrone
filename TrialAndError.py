import numpy as np

# Beispiel-Array (n x 3 x 2)
array = np.array([[[5, 1], [2, 1], [3, 4]],
                  [[0, 2], [5, 7], [1, 1]],
                  [[8, 6], [9, 3], [4, 2]],
                  [[3, 9], [7, 8], [2, 3]]])

# Extrahiere die erste Spalte der ersten Matrix (n x 1)
first_column = array[:, 0, 0]

# Bestimme den Index der ersten 0 in der ersten Spalte
zero_index = np.where(first_column == 0)[0]

# Wenn es keine 0 gibt, sortiere den gesamten Array
if len(zero_index) == 0:
    sorted_array = array[np.argsort(first_column)]
else:
    # Sortiere nur bis zum ersten Auftreten von 0
    sorted_array = array[:zero_index[0]]
    sorted_array = sorted_array[np.argsort(sorted_array[:, 0, 0])]

print("Sortierter Array:")
print(sorted_array)



array = np.array([
                  [5, [1, 2],  [1, 1],  [4, 5]],
                  [0, [2, 3],  [7, 8],  [1, 1]],
                  [8, [6, 7],  [3, 4],  [2, 3]],
                  [3, [9, 0],  [8, 9],  [3, 4]]], dtype=object)

itera=iter(array)
for i in itera:
    print(i[1][0])

