"""
Train and test functions.
"""
import os
import functools
from sys import platform
from timeit import timeit
from classification.series_classifier import SeriesClassifier
from preprocessing.create_class_models import create_class_models
import preprocessing.cmeans_clustering as cmeans_clustering
import preprocessing.read_data as read_data
from classification.binary_classifier_model import BinaryClassifierModel
import params
import classification.binary_classifier_factory  

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
    
    create_classifiers_fun = {
            0: classification.binary_classifier_factory.create_asymetric_binary_classifiers,
            1: classification.binary_classifier_factory.create_symetric_binary_classifiers,
            2: classification.binary_classifier_factory.create_k_vs_all_binary_classifiers,
        }[params.method]
    binary_classifier_models = create_classifiers_fun(class_models, previous_considered_indices, move)
    for model in binary_classifier_models:
        model.train()

    res = SeriesClassifier(class_models, binary_classifier_models)
    return res


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
