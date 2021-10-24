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


def _process_file(file_info, prev):
    file_path = file_info["path"]
    series = read_data.process_data(file_path)
    fcm = fcm_creation.create_fcm(series, prev)
    print(f"Processing {file_info['name']} from class {file_info['class']} ended\n")
    return file_info["class"], fcm


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

    file_infos = []
    print(datetime.datetime.now())

    for class_dir in (entry for entry in os.scandir(train_dir) if entry.is_dir()):
        for file in os.scandir(class_dir.path):
            if file.name.endswith(".csv"):
                file_infos.append(
                    {"class": int(class_dir.name), "path": file.path, "name": file.name}
                )

    with Pool() as p:
        fun = functools.partial(_process_file, prev=prev)
        #W tym momencie jest :8 tylko dla testowania
        svm_training_data = p.map(fun, file_infos[:8])
    svm_classes, svm_fcms = zip(*svm_training_data)
    print(svm_classes)
    print(svm_fcms)


if __name__ == "__main__":
    train(os.path.join("UWaveGestureLibrary_Preprocessed", "Train"))
