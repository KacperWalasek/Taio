"""
Module for training.
"""
import os
import datetime
import functools
from multiprocessing import Pool
import numpy as np
import read_data
import fcm_creation


def _process_single_file(file, class_info, prev):
    file_path = os.path.join(class_info["dir"], file)
    series = read_data.process_data(file_path)
    fcm = fcm_creation.create_fcm(series, prev)
    print(f"Processing {file} from class {class_info['number']} ended\n")
    return fcm


def _process_class(class_info, prev):
    fun = functools.partial(_process_single_file, class_info=class_info, prev=prev)
    #Na razie jest :5 poniżej do debugowania tylko
    return class_info["number"], list(map(fun, class_info["files"][:5]))


def train(train_dir):
    """
    Train classifier.

    Parameters:
    ----------
    train_dir : string
        Path to Train directory which contains directories (named with class numbers)
        with CSV time series files.
    """
    prev = np.r_[1, 2, 3, 4]

    classes = []
    print(datetime.datetime.now())

    for class_dir in (entry for entry in os.scandir(train_dir) if entry.is_dir()):
        classes.append(
            {
                "number": int(class_dir.name),
                "dir": class_dir.path,
                "files": [
                    file for file in os.listdir(class_dir.path) if file.endswith(".csv")
                ],
            }
        )

    with Pool() as p:
        fun = functools.partial(_process_class, prev=prev)
        svm_training_data = p.map(fun, classes)
        svm_training_data = [
            (number, fcm) for (number, fcms) in svm_training_data for fcm in fcms
        ]
        svm_classes, svm_fcms = zip(*svm_training_data)
        print(svm_classes)
        print(svm_fcms)


if __name__ == "__main__":
    train(os.path.join("..", "UWaveGestureLibrary_Preprocessed", "Train"))
