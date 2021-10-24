"""This file contains all functionalities associated with svm classification"""

from sklearn import svm
import joblib
import numpy as np


def create_and_train_svm(model_array, labels, **kwargs):
    """
    Takes
    - numpy array of model parameters of shape (n_samples,3,3)
    - label array of the same size as training data
    - named arguments to describe svm (more details in:
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
    Creates SVM algorithm and trains using training_data.
    Returns trained algorithm
    """
    training_data = model_array.reshape(model_array.shape[0], -1)
    print("Train SVM algorithm...")
    return svm.SVC(**kwargs).fit(training_data, labels)


def save_svm(svm_algorithm, path):
    """Saves svm algorithm to file"""
    joblib.dump(svm_algorithm, path)


def load_svm(path):
    """Loads and returns svm from file of given path"""
    return joblib.load(path)


def classify(svm_algorithm, model_array):
    """
    Takes svm algorithm and model_array of shape (n_samples,3,3).
    Predicts class of each element of model_array.
    Returns list of predictions
    """
    data = model_array.reshape(model_array.shape[0], -1)
    print("Classify models...")
    return svm_algorithm.predict(data)


if __name__ == "__main__":
    train = np.array([np.ones((3, 3)), np.zeros((3, 3))])
    save_svm(
        create_and_train_svm(train, np.array([0, 1]), kernel="rbf", C=100, gamma=0.01),
        "./svmFile.joblib",
    )
    test = np.array([
        np.array([1, 0, 3]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0])
    ])
    result = classify(
        load_svm("./svmFile.joblib"),
        np.array([np.ones((3, 3)), np.zeros((3, 3)), test]),
    )
    print("Classification result: ", result)
