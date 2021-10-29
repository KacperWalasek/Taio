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


def _process_file(file_info, length_percent, prev, split):
    file_path = file_info["path"]
    series = read_data.process_data(file_path, length_percent)
    fcms = np.array(fcm_creation.create_fcms(series, prev, 4, split))
    return file_info["class"], fcms


def _create_models(dir_path, length_percent, previous_considered_indices, split):
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
    split : bool
        If True, we create FCM for each coordinate of time series.
        If False, we create one FCM.

    Returns
    -------
    zip
        Zip object containing class name and FCM model.

    """
    prev = np.array(previous_considered_indices)

    class_list = []
    series_list = []

    for class_dir in (entry for entry in os.scandir(dir_path) if entry.is_dir()):
        for file in os.scandir(class_dir.path):
            if file.name.endswith(".csv"):
                class_list.append(int(class_dir.name))
                series_list.append(read_data.process_data(file.path, length_percent))
    
    fcm_list = fcm_creation.create_fcms(series_list, prev, 20, split)

    return class_list, fcm_list

    #To niżej chyba do wywalenia
    with Pool() as p:
        fun = functools.partial(
            _process_file, length_percent=length_percent, prev=prev, split=split
        )
        svm_training_data = p.map(fun, file_infos)
    return zip(*svm_training_data)


def train(dir_path, length_percent, previous_considered_indices, split):
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
    split : bool
        If True, we create FCM for each coordinate of time series.
        If False, we create one FCM.

    Returns
    -------
    None.

    """
    print("Train")
    print("Create FCM models...")
    classes, fcms = _create_models(
        dir_path, length_percent, previous_considered_indices, split
    )
    print("Train SVM algorithm...")
    svm.save_svm(
        svm.create_and_train_svm(np.array(fcms), np.array(classes), kernel="rbf"),
        "./svmFile.joblib",
    )


def test(dir_path, length_percent, previous_considered_indices, split):
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
    split : bool
        If True, we create FCM for each coordinate of time series.
        If False, we create one FCM.

    Returns
    -------
    number
        Classification result.

    """
    print("Test")
    print("Create FCM models...")
    classes, fcms = _create_models(
        dir_path, length_percent, previous_considered_indices, split
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


if __name__ == "__main__":
    file_description = {
        "class": 4,
        "path": "UWaveGestureLibrary_Preprocessed/Train/4/740.csv",
        "name": "740.csv",
    }
    _process_file(file_description, 0.5, np.r_[1, 2], True)
