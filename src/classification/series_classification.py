"""
Train and test functions.
"""
import os
import functools
from sys import platform
from timeit import timeit
from classification.binary_classifier_model import BinaryClassifierModel
from classification.series_classifier import SeriesClassifier
from preprocessing.create_class_models import create_class_models
import preprocessing.cmeans_clustering as cmeans_clustering
import preprocessing.read_data as read_data

def train(dir_path, previous_considered_indices, move, concept_count):
    """
    Parameters
    ----------
    dir_path : string
    previous_considered_indices : list
    move : int

    Returns
    -------
    SeriesClassifier

    """

    class_models = create_class_models(dir_path, previous_considered_indices, concept_count)
    
    binary_classifier_models = []

    for model1_idx, model1 in enumerate(class_models):
        for model2_idx, model2 in enumerate(class_models):
            if model1 != model2:
                model2_memberships = cmeans_clustering.find_memberships(
                    model2[1], model1[2][0]
                )
                binary_classifier_models.append(
                    BinaryClassifierModel(
                        (model1_idx, model2_idx),
                        (model1[2][1], model2_memberships),
                        model1[2][0],
                        previous_considered_indices,
                        move,
                    )
                )
                
    binary_classifier_models = map(_binary_model_train_wrapper,binary_classifier_models)

    res = SeriesClassifier(class_models, binary_classifier_models)
    return res


def _binary_model_train_wrapper(model):
    return model.train()


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
                class_list.append(class_dir.name)

    predicted_classes = series_classifier.predict(series_list)

    result = sum(predicted_classes == class_list) / len(series_list)

    return result
