"""
Train and test functions.
"""
import os
import train_model
import test_series
import class_model
import read_data


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
    tuple(list(tuple(int, int, numpy.ndarray, numpy.ndarray)), int)

    """

    class_dirs = [entry for entry in os.scandir(dir_path) if entry.is_dir()]

    class_models = []

    for class_dir in class_dirs:
        class_models.append(
            class_model.ClassModel(
                int(class_dir.name),
                class_dir.path,
                length_percent,
                previous_considered_indices,
                move,
            )
        )

    for model in class_models:
        model.start()

    for model in class_models:
        model.join()

    train_models = []

    for model1 in class_models:
        for model2 in class_models:
            if model1 != model2:
                classes_series_lists = [
                    # tu series_list jest puste
                    [model1.class_number, model1.series_list],
                    [model2.class_number, model2.series_list],
                ]
                train_models.append(
                    train_model.TrainModel(
                        classes_series_lists,
                        model1.clusters,
                        length_percent,
                        previous_considered_indices,
                        move,
                    )
                )

    for model in train_models:
        model.start()

    for model in train_models:
        model.join()

    models = [
        (model.class1, model.class2, model.clusters, model.matrices)
        for model in train_models
    ]

    return models, len(class_dirs)


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
