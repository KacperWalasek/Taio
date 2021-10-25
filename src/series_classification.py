"""
Module for testing.
"""
import os
import datetime
import functools
from multiprocessing import Pool
import numpy as np
import read_data
import fcm_creation
import svm
import sys

def _process_file(file_info, prev):
    file_path = file_info["path"]
    series = read_data.process_data(file_path)
    fcm = fcm_creation.create_fcm(series, prev)
    print(f"Processing {file_info['name']} from class {file_info['class']} ended\n")
    return file_info["class"], fcm


def create_models(dir):
    """
    Creates fcm models out of files in given directory

    Parameters:
    ----------
    dir : string
        Path to directory which contains directories (named with class numbers)
        with CSV time series files.
    """
    prev = np.r_[1, 2, 3, 4]

    file_infos = []
    print(datetime.datetime.now())

    for class_dir in (entry for entry in os.scandir(dir) if entry.is_dir()):
        for file in os.scandir(class_dir.path):
            if file.name.endswith(".csv"):
                file_infos.append(
                    {"class": int(class_dir.name), "path": file.path, "name": file.name}
                )

    with Pool() as p:
        fun = functools.partial(_process_file, prev=prev)
        svm_training_data = p.map(fun, file_infos)
    return zip(*svm_training_data)


def train(dir):
    svm_classes, svm_fcms = create_models(dir)
    svm.save_svm(
        svm.create_and_train_svm(np.array(svm_fcms), np.array(svm_classes), kernel="rbf"),
        "./svmFile.joblib",
    )


def test(dir):
    svm_classes, svm_fcms = create_models(dir)
    classes = svm.classify(svm.load_svm("./svmFile.joblib"),np.array(svm_fcms))
    correct_ones = 0
    for i in range(len(classes)):
        if classes[i] == svm_classes[i]:
            correct_ones += 1
    print("Classification result: ",correct_ones/len(classes))


if __name__ == "__main__":
    if len(sys.argv)!=2 or ( sys.argv[1]!='train' and sys.argv[1]!='test' ):
        print('\'train\' or \'test\' argument is needed')
    else:
        if sys.argv[1]=='train':
            train(os.path.join("UWaveGestureLibrary_Preprocessed", "Train"))
        else:
            test(os.path.join("UWaveGestureLibrary_Preprocessed", "Test"))

