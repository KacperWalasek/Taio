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

available_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

if platform in ["linux", "linux2"]:
    from ray.util.multiprocessing import Pool
    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=available_cpus)
else:
    from multiprocessing import Pool


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

    class_dirs = [
        (entry.name, entry.path) for entry in os.scandir(dir_path) if entry.is_dir()
    ]

    fun = functools.partial(_read_and_cluster_class_series, concept_count=concept_count)
    with Pool(min(available_cpus, len(class_dirs))) as p:
        class_models = p.map(fun, class_dirs)

    too_short_series = next(
        (
            (series, class_model[0])
            for class_model in class_models
            for series in class_model[1]
            if series.shape[1] < max(previous_considered_indices) + 1
        ),
        None,
    )
    if too_short_series is not None:
        raise RuntimeError(
            "Specified previous_considered_indices array invalid for given dataset"
            f" - one of the series in class {too_short_series[1]}"
            " after preprocessing with read_data module has only"
            f" {too_short_series[0].shape[1] + 1} elements."
            " Remember that c-means processed series has one element less"
            " due to adding adjacent differences to points (the first one is discarded)."
        )
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

    with Pool(min(available_cpus, len(binary_classifier_models))) as p:
        binary_classifier_models = p.map(
            _binary_model_train_wrapper, binary_classifier_models
        )

    res = SeriesClassifier(class_models, binary_classifier_models)
    return res


def _binary_model_train_wrapper(model):
    return model.train()


def _read_and_cluster_class_series(class_dir, concept_count):
    series_list = []
    for file in os.scandir(class_dir[1]):
        if file.name.endswith(".csv"):
            series_list.append(read_data.process_data(file.path, 1))

    return (
        class_dir[0],
        series_list,
        cmeans_clustering.find_clusters(series_list, concept_count),
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
                class_list.append(class_dir.name)

    predicted_classes = series_classifier.predict(series_list)

    result = sum(predicted_classes == class_list) / len(series_list)

    return result
