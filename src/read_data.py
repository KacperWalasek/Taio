"""
Module for reading data from CSV file,
deleting first and last redundant rows
and normalising data.
"""
import numpy as np


def process_data(file, length):
    """
    Read data from CSV file and prepare them for next steps.
    Preparation consists of deleting redundant rows from beginning
    and end of file and normalising data to [0, 1] interval.

    Parameters
    ----------
    file : string
        Path to CSV file which has data that are supposed to be processed.
    length : number
        Percent of rows of time series which should be taken into consideration.

    Returns
    -------
    numpy.ndarray
        Two dimensional numpy array where each row from
        file is a one dimensional array.

    """

    data = np.genfromtxt(file, delimiter=",")

    data = data[: int(length * data.shape[0])]

    data = _delete_redundant_rows(data)

    data = _normalize(data)

    return data


def _delete_redundant_rows(data):
    """
    Delete first n-1 rows when first n rows are the same
    (last rows analogously).

    Parameters
    ----------
    data : numpy.ndarray
        Two dimensional numpy array which will be processed.

    Returns
    -------
    numpy.ndarray
        Two dimensional numpy array.

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


def _normalize(data):
    """
    Normalize data (all elements together) to [0, 1] interval.

    Parameters
    ----------
    data : numpy.ndarray
        Two dimensional numpy array which will be processed.

    Returns
    -------
    numpy.ndarray
        Two dimensional numpy array normalized to [0, 1] interval.

    """

    norm = np.linalg.norm(data, axis=1).max()
    return (data / norm + 1) / 2
