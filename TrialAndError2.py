import numpy as np

# Your 2D object array
arr = np.array([[np.array([1, 2]), 5], [np.array([3, 4]), 2], [np.array([2, 3]), 4]], dtype=object)

# Sort the array based on the first element (the vector) in each row
arr_sorted = sorted(arr, key=lambda x: x[0].tolist())  # Convert the numpy array to a list for comparison

# Convert back to a numpy array if needed
arr_sorted = np.array(arr_sorted, dtype=object)

print(arr_sorted)
