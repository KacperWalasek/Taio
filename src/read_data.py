"""Module for reading data from csv file, deleting first and last redundant rows
and normalising data"""
import numpy as np


def process_data(file):
    """Read data from file and prepare them for next steps"""

    data = np.genfromtxt(file, delimiter=",")

    data = delete_redundant_rows(data)

    data = normalize(data)

    return data


def delete_redundant_rows(data):
    """Delete redundant rows - deletes first n-1 rows when first n rows are the same
    (last rows analogously)"""

    first = data[0]
    i = 1

    while i < len(data) and np.array_equal(first, data[i]):
        i = i + 1

    data = data[i - 1 :]

    last = data[-1]
    i = 2

    while i <= len(data) and np.array_equal(last, data[-i]):
        i = i + 1

    if i == 2:
        return data

    return data[: -(i - 2)]


def normalize(data):
    """Normalize data - all elements together"""

    norm = np.linalg.norm(data)
    return data / norm
