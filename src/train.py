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
    prev = np.r_[1, 2, 3, 4]

    classes = []
    print(datetime.datetime.now())

    for class_dir in (entry for entry in os.scandir(train_dir) if entry.is_dir()):
        classes.append(
            {
                "class_number": int(class_dir.name),
                "dir": class_dir.path,
                "files": [
                    file for file in os.listdir(class_dir.path) if file.endswith(".csv")
                ],
            }
        )

    for i in range(len(classes[0]["files"])):
        for cls in classes:
            if len(cls["files"]) <= i:
                continue
            file_name = os.path.join(cls["dir"], cls["files"][i])
            series = read_data.process_data(file_name)
            print("class", cls["class_number"])
            fcm_creation.create_fcm(series, prev)


if __name__ == "__main__":
    train(os.path.join("src", "UWaveGestureLibrary_Preprocessed", "Train"))
