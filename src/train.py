"""
Module for training.
"""

import os
import datetime
import numpy as np
import read_data
import fcm_creation


def train(train_dir):
    """
    Train classifier.

    Parameters:
    ----------
    train_dir : string
        Path to Train directory which contains directories (named with class numbers)
        with CSV time series files.
    """
    prev = [1, 2, 3, 4]

    i = 0
    classes = []
    print(datetime.datetime.now())

    for dir_name, _, files in os.walk(train_dir):
        if i == 0:
            i = i + 1
            continue

        classes.append(
            {"class_number": int(dir_name[-1]), "dir": dir_name, "files": files}
        )

    for i in range(len(classes[0]["files"])):
        for cls in classes:
            if len(cls["files"]) <= i:
                continue
            file_name = os.path.join(cls["dir"], cls["files"][i])

            series = read_data.process_data(file_name)
            print("class", cls["class_number"])
            fcm_creation.create_fcm(series, np.array(prev))


if __name__ == "__main__":
    train("src/UWaveGestureLibrary_Preprocessed/Train")
