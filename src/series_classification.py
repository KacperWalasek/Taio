"""
This module contains train and test functions
which should be used to train classifier and test it.
"""
import os
import functools
from multiprocessing import Pool
import numpy as np
import read_data
import fcm_creation
import svm


def _process_file(file_info, length, prev):
    file_path = file_info["path"]
    series = read_data.process_data(file_path, length)
    fcm = fcm_creation.create_fcm(series, prev)
    return file_info["class"], fcm


def _create_models(dir_path, length_percent, previous_considered_indices):
    """
    Creates fcm models out of files in given directory.

    Parameters
    ----------
    dir_path : string
        Path to directory which contains directories (named with class numbers)
        with CSV time series files.
    length_percent : number
        Percent of rows of time series which should be taken into consideration.
    previous_considered_indices : list
        List containing indices of previous elements which will be
        FCM's input for predicting the next one. For example, if you want
        to predict next element using the current one, the previous one and
        before the previous one you can pass numbers: 1, 2, 3.
        This argument's entries do not have to be sorted.

    Returns
    -------
    iterator of tuples
        Iterator of tuples containing class name and FCM model.

    """
    prev = np.array(previous_considered_indices)

    file_infos = []

    for class_dir in (entry for entry in os.scandir(dir_path) if entry.is_dir()):
        for file in os.scandir(class_dir.path):
            if file.name.endswith(".csv"):
                file_infos.append(
                    {"class": int(class_dir.name), "path": file.path, "name": file.name}
                )

    with Pool() as p:
        fun = functools.partial(_process_file, length=length_percent, prev=prev)
        svm_training_data = p.map(fun, file_infos)
    return zip(*svm_training_data)


def train(dir_path, length_percent, previous_considered_indices):
    """
    Trains svm classifier.

    Parameters
    ----------
    dir_path : string
        Path to directory which contains directories (named with class numbers)
        with CSV time series files.
    length_percent : number
        Percent of rows of time series which should be taken into consideration.
    previous_considered_indices : list
        List containing indices of previous elements which will be
        FCM's input for predicting the next one. For example, if you want
        to predict next element using the current one, the previous one and
        before the previous one you can pass numbers: 1, 2, 3.
        This argument's entries do not have to be sorted.

    Returns
    -------
    None.

    """
    print("Train")
    print("Create FCM models...")
    classes, fcms = _create_models(
        dir_path, length_percent, previous_considered_indices
    )
    print("Train SVM algorithm...")
    svm.save_svm(
        svm.create_and_train_svm(np.array(fcms), np.array(classes), kernel="rbf"),
        "./svmFile.joblib",
    )


def test(dir_path, length_percent, previous_considered_indices):
    """
    Tests svm classifier.

    Parameters
    ----------
    dir_path : string
        Path to directory which contains directories (named with class numbers)
        with CSV time series files.
    length_percent : number
        Percent of rows of time series which should be taken into consideration.
    previous_considered_indices : list
        List containing indices of previous elements which will be
        FCM's input for predicting the next one. For example, if you want
        to predict next element using the current one, the previous one and
        before the previous one you can pass numbers: 1, 2, 3.
        This argument's entries do not have to be sorted.

    Returns
    -------
    number
        Classification result.

    """
    print("Test")
    print("Create FCM models...")
    classes, fcms = _create_models(
        dir_path, length_percent, previous_considered_indices
    )
    print("Classify time series...")
    predicted_classes = svm.classify(svm.load_svm("./svmFile.joblib"), np.array(fcms))
    correct_ones = 0
    for predicted_class_number, class_number in zip(predicted_classes, classes):
        if predicted_class_number == class_number:
            correct_ones += 1
    result = correct_ones / len(classes)
    print("Classification result: ", result)
    return result
