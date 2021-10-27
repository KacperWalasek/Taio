"""
This file contains all functionalities associated with SVM classification.
"""
from sklearn import svm
import numpy as np
import joblib


def create_and_train_svm(model_array, labels, **kwargs):
    """
    Creates SVM algorithm and trains using training_data.

    Parameters
    ----------
    model_array : numpy.ndarray
        Array of arrays of model parameters of shape (n_samples, 1/3, c, c).
    labels : list
        List of class labels of the same size as training data.
    **kwargs
        Named arguments to describe SVM (more details in:
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC).

    Returns
    -------
    object
        Trained algorithm.

    """
    training_data = model_array.reshape(model_array.shape[0], -1)
    return svm.SVC(**kwargs).fit(training_data, labels)


def save_svm(svm_algorithm, path):
    """
    Saves SVM algorithm to file.

    Parameters
    ----------
    svm_algorithm : object
        SVM algorithm to be saved.
    path : string
        Path to save algorithm to.

    Returns
    -------
    None.

    """
    joblib.dump(svm_algorithm, path)


def load_svm(path):
    """
    Loads and returns SVM algorithm from file of given path.

    Parameters
    ----------
    path : string
        Path to saved algorithm.

    Returns
    -------
    object
        SVM algorithm.

    """
    return joblib.load(path)


def classify(svm_algorithm, model_array):
    """
    Predicts class of each element of model_array.

    Parameters
    ----------
    svm_algorithm : object
        SVM algorithm.
    model_array : numpy.ndarray
        Array of arrays of model parameters of shape (n_samples, 1/3, c, c).

    Returns
    -------
    numpy.ndarray
        Predictions.

    """
    data = model_array.reshape(model_array.shape[0], -1)
    return svm_algorithm.predict(data)


if __name__ == "__main__":
    train = np.array([np.ones((3, 3)), np.zeros((3, 3))])
    save_svm(
        create_and_train_svm(train, np.array([0, 1]), kernel="rbf", C=100, gamma=0.01),
        "./svmFile.joblib",
    )
    test = np.array([np.array([1, 0, 3]), np.array([0, 0, 0]), np.array([0, 0, 0])])
    result = classify(
        load_svm("./svmFile.joblib"),
        np.array([np.ones((3, 3)), np.zeros((3, 3)), test]),
    )
    print("Classification result: ", result)
