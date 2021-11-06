"""
Train and test functions.
"""
import os
import train_model
import test_series
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
    tuple(list(tuple(int, int, numpy.ndarray)), int)

    """

    train_models = []
    class_dirs = [entry for entry in os.scandir(dir_path) if entry.is_dir()]

    for i in range(len(class_dirs)):
        for j in range(i + 1, len(class_dirs)):
            classes_paths = [
                [int(class_dirs[i].name), class_dirs[i].path],
                [int(class_dirs[j].name), class_dirs[j].path],
            ]
            train_models.append(
                train_model.TrainModel(
                    classes_paths, length_percent, previous_considered_indices, move
                )
            )

    for model in train_models:
        model.start()

    for model in train_models:
        model.join()

    models = [(model.class1, model.class2, model.matrices) for model in train_models]

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
    models : list(tuple(int, int, numpy.ndarray))
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
