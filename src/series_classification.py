"""
This module contains train and test functions
which should be used to train classifier and test it.
"""
import os
import numpy as np
import read_data
import fcm_creation
import svm


def _get_series(dir_path, length_percent):
    """
    Returns list of time series from given directory.

    Parameters
    ----------
    dir_path : string
        Path to directory which contains directories (named with class numbers)
        with CSV time series files.
    length_percent : number
        Percent of rows of time series which should be taken into consideration.

    Returns
    -------
    tuple
        Tuple containing list of class names and list of time series.

    """
    class_list = []
    series_list = []

    for class_dir in (entry for entry in os.scandir(dir_path) if entry.is_dir()):
        for file in os.scandir(class_dir.path):
            if file.name.endswith(".csv"):
                class_list.append(int(class_dir.name))
                series_list.append(read_data.process_data(file.path, length_percent))

    return class_list, series_list


def _create_training_models(
    dir_path, length_percent, previous_considered_indices, split
):
    """
    Creates FCM models out of files in given directory.

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
    tuple
        Tuple containing list with class names, list with FCM models and list of cluster centers.

    """
    class_list, series_list = _get_series(dir_path, length_percent)

    cluster_centers, fcm_list = fcm_creation.create_training_fcms(
        series_list, np.array(previous_considered_indices), 20, split
    )
    return class_list, fcm_list, cluster_centers


def _create_test_models(
    dir_path, length_percent, previous_considered_indices, split, cluster_centers
):
    """
    Creates FCM models out of files in given directory.

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
    cluster_centers : list
        List containing cluster centers (numpy.ndarray).
        If split is True, then list contains cluster centers for every coordinate of time series.
        If split is False, then list contains only one numpy.ndarray.

    Returns
    -------
    tuple
        Tuple containing list with class names, list with FCM models.

    """
    class_list, series_list = _get_series(dir_path, length_percent)

    fcm_list = fcm_creation.create_test_fcms(
        series_list, np.array(previous_considered_indices), 20, split, cluster_centers
    )
    return class_list, fcm_list


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
    list
        List containing cluster centers (numpy.ndarray).
        If split is True, then list contains cluster centers for every coordinate of time series.
        If split is False, then list contains only one numpy.ndarray.

    """
    print("Train", flush=True)
    classes, fcms, cluster_centers = _create_training_models(
        dir_path, length_percent, previous_considered_indices, split
    )

    print("Train SVM algorithm...", flush=True)
    svm.save_svm(
        svm.create_and_train_svm(np.array(fcms), np.array(classes), kernel="rbf"),
        "./svmFile.joblib",
    )

    return cluster_centers


def test(dir_path, length_percent, previous_considered_indices, split, cluster_centers):
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
    cluster_centers : list
        List containing cluster centers (numpy.ndarray).
        If split is True, then list contains cluster centers for every coordinate of time series.
        If split is False, then list contains only one numpy.ndarray.

    Returns
    -------
    number
        Classification result.

    """
    print("Test", flush=True)
    print("Create FCM models...", flush=True)
    classes, fcms = _create_test_models(
        dir_path, length_percent, previous_considered_indices, split, cluster_centers
    )

    print("Classify time series...", flush=True)
    predicted_classes = svm.classify(svm.load_svm("./svmFile.joblib"), np.array(fcms))

    correct_ones = 0
    for predicted_class_number, class_number in zip(predicted_classes, classes):
        if predicted_class_number == class_number:
            correct_ones += 1
    result = correct_ones / len(classes)
    print("Classification result: ", result, flush=True)
    return result
