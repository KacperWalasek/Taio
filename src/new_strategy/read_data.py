"""
Module for reading data from CSV file,
deleting first and last redundant rows.
"""
import numpy as np


def process_data(file, length_percent):
    """
    Parameters
    ----------
    file : string
    length_percent : number
        [0, 1]

    Returns
    -------
    numpy.ndarray

    """

    data = np.genfromtxt(file, delimiter=",")

    data = _delete_redundant_rows(data)

    data = data[: int(length_percent * data.shape[0])]

    return data.T


def _delete_redundant_rows(data):
    """
    Parameters
    ----------
    data : numpy.ndarray

    Returns
    -------
    numpy.ndarray

    """

    # first rows
    first = data[0]
    i = 1

    while i < len(data) and np.array_equal(first, data[i]):
        i = i + 1

    data = data[i - 1 :]

    # last rows
    last = data[-1]
    i = 2

    while i <= len(data) and np.array_equal(last, data[-i]):
        i = i + 1

    if i == 2:
        return data

    return data[: -(i - 2)]
