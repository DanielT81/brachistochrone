import numpy as np
import pandas as pd

arr = np.zeros([3,4], dtype=object)

arr[1,1] = [0,9]

print(arr[1,1])