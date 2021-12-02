"""
Preprocess data delivered in *.arff format.
"""

import sys
import os
from shutil import rmtree
from scipy.io.arff import loadarff
import numpy as np


def preprocess_data(name, nr_of_coordinates):
    """
    Preprocess data delivered in *.arff format.

    Parameters
    ----------
    name : string
        name of dataset
    nr_of_coordinates : int
        number of dimensions in time series data

    Returns
    -------
    None.

    """

    os.chdir(name)
    _preprocess_data_to_single_directory(
        name, "_TRAIN.arff", "Train", nr_of_coordinates
    )
    _preprocess_data_to_single_directory(name, "_TEST.arff", "Test", nr_of_coordinates)


def _preprocess_data_to_single_directory(name, suffix, directory, nr_of_coordinates):
    """
    Preprocess data delivered in *.arff format.

    Parameters
    ----------
    name : string
        name of dataset
    suffix : string
        suffix of *.arff file to be considered
    directory : string
        name of directory to create and put CSV files into
    nr_of_coordinates : int
        number of dimensions in time series data

    Returns
    -------
    None.

    """

    if os.path.exists(directory):
        rmtree(directory, ignore_errors=True)
    os.mkdir(directory)

    data = []

    if nr_of_coordinates == 1:
        with open(name + suffix, "r", encoding="UTF-8") as file:
            data.append(loadarff(file)[0])
    else:
        for i in range(nr_of_coordinates):
            with open(
                name + "Dimension" + str(i + 1) + suffix, "r", encoding="UTF-8"
            ) as file:
                data.append(loadarff(file)[0])

    for i in range(len(data[0])):
        series_length = len(data[0][i]) - 1
        file_data = np.empty([series_length, nr_of_coordinates])

        for j in range(nr_of_coordinates):
            file_data[:, j] = np.asarray(data[j][i].item()[0:series_length])

        dir_path = os.path.join(directory, data[0][i][-1].decode("UTF-8"))

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        file_path = os.path.join(dir_path, str(i) + ".csv")

        np.savetxt(file_path, file_data, delimiter=",")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise RuntimeError("Too few arguments")

    preprocess_data(sys.argv[1], int(sys.argv[2]))
