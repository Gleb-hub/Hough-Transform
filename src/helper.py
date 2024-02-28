import numpy as np


def find_max(arr):
    max_value = np.max(arr)
    max_index_flat = np.argmax(arr)
    row, col = np.unravel_index(max_index_flat, arr.shape)
    return max_value, (row, col)
