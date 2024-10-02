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



coordinates = [(1, 2), (3, 4), (5, 6)]  # Beispielhafte Liste von Koordinaten
index = 0

while True:
    try:
        # Versuch, auf eine Koordinate zuzugreifen
        x, y = coordinates[index]
        # Codestück A (wenn Koordinate existiert)
        print(f"Koordinate {index}: x = {x}, y = {y}")
    except IndexError:
        # Codestück B (wenn Koordinate nicht existiert)
        print(f"Keine Koordinate an Index {index}. Beende Schleife.")
        break  # Schleife beenden oder andere Aktion ausführen

    index += 1
