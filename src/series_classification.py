"""
Train and test functions.
"""
import os
from binary_classifier_model import BinaryClassifierModel
from class_model import ClassModel
from series_classifier import SeriesClassifier
import read_data
import cmeans_clustering

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

    class_dirs = [entry for entry in os.scandir(dir_path) if entry.is_dir()]

    class_models = []

    for class_dir in class_dirs:
        class_models.append(
            ClassModel(
                int(class_dir.name),
                class_dir.path,
                length_percent
            )
        )

    for model in class_models:
        model.run() # tu było start wcześniej ~Kacper

    # for model in class_models:
    #     model.join()

    binary_classifier_models = []

    for model1 in class_models:
        for model2 in class_models:
            if model1 != model2:
                model2_memberships = cmeans_clustering.find_memberships(model2.series_list, model1.centroids)
                binary_classifier_models.append(
                    BinaryClassifierModel(
                        (model1.class_number, model2.class_number),
                        (model1.memberships, model2_memberships),
                        model1.centroids,
                        previous_considered_indices,
                        move,
                    )
                )

    for model in binary_classifier_models:
        model.train()

    return SeriesClassifier(class_models, binary_classifier_models)


def test(
    dir_path, length_percent, previous_considered_indices, move, models, class_count
):
    """
    Parameters
    ----------
    dir_path : string
    length_percent : number
        [0, 1]
    previous_considered_indices : list
    move : int
    models : list(tuple(int, int, numpy.ndarray, numpy.ndarray))
    class_count : int

    Returns
    -------
    number
        [0, 1]

    """
    tests = []

    for class_dir in (entry for entry in os.scandir(dir_path) if entry.is_dir()):
        for file in os.scandir(class_dir.path):
            if file.name.endswith(".csv"):
                tests.append(
                    test_series.Test(
                        int(class_dir.name),
                        read_data.process_data(file.path, length_percent),
                        previous_considered_indices,
                        move,
                        models,
                        class_count,
                    )
                )

    for series_test in tests:
        series_test.start()

    for series_test in tests:
        series_test.join()

    result = sum([series_test.result.value for series_test in tests]) / len(tests)

    return result
