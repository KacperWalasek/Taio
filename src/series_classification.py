"""
Train and test functions.
"""
import os
import functools
from sys import platform
from binary_classifier_model import BinaryClassifierModel
from series_classifier import SeriesClassifier
import read_data
import cmeans_clustering

if platform in ["linux", "linux2"]:
    from ray.util.multiprocessing import Pool
else:
    from multiprocessing import Pool


def train(dir_path, length_percent, previous_considered_indices, move):
    """
    Parameters
    ----------
    dir_path : string
    length_percent : number
        [0, 1]
    previous_considered_indices : list
    move : int

    Returns
    -------
    SeriesClassifier

    """

    class_dirs = [
        (entry.name, entry.path) for entry in os.scandir(dir_path) if entry.is_dir()
    ]

    fun = functools.partial(_clustering, length_percent=length_percent)
    with Pool() as p:
        class_models = p.map(fun, class_dirs)

    binary_classifier_models = []

    for model1 in class_models:
        for model2 in class_models:
            if model1 != model2:
                model2_memberships = cmeans_clustering.find_memberships(
                    model2[1], model1[2][0]
                )
                binary_classifier_models.append(
                    BinaryClassifierModel(
                        (model1[0], model2[0]),
                        (model1[2][1], model2_memberships),
                        model1[2][0],
                        previous_considered_indices,
                        move,
                    )
                )

    for model in binary_classifier_models:
        model.train()

    return SeriesClassifier(class_models, binary_classifier_models)


def _clustering(class_dir, length_percent):
    series_list = []
    for file in os.scandir(class_dir[1]):
        if file.name.endswith(".csv"):
            series_list.append(read_data.process_data(file.path, length_percent))

    return (
        int(class_dir[0]),
        series_list,
        cmeans_clustering.find_clusters(series_list, 12),
    )


def test(dir_path, length_percent, series_classifier):
    """
    Parameters
    ----------
    dir_path : string
    length_percent : number
        [0, 1]
    series_classifier : SeriesClassifier

    Returns
    -------
    number
        [0, 1]

    """
    series_list = []
    class_list = []

    for class_dir in (entry for entry in os.scandir(dir_path) if entry.is_dir()):
        for file in os.scandir(class_dir.path):
            if file.name.endswith(".csv"):
                series_list.append(read_data.process_data(file.path, length_percent))
                class_list.append(int(class_dir.name))

    predicted_classes = series_classifier.predict(series_list)

    result = sum(predicted_classes == class_list) / len(series_list)

    return result
