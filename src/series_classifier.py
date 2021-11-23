"""
Class for classifying list of time series.
"""

import pickle
import numpy as np
import cmeans_clustering as cmeans


class SeriesClassifier:
    """
    A class for classifying time series based upon trained binary classifiers.

    Parameters
    ----------
    class_models : list
        A list of class models.
    binary_classifiers : list
        A list of previously trained BinaryClassifier objects corresponding to class_models.

    """

    def __init__(self, class_models, binary_classifiers):
        self.class_models = list(map(lambda x: (x[0], x[2][0]), class_models))
        self._binary_classifiers = binary_classifiers

    def predict(self, series_list):
        """
        This method predicts class number values.

        Parameters
        ----------
        series_list : list
            A list of series. Each series is represented as numpy.ndarray
            of shape (space_dimension, length_of_series).

        Returns
        -------
        numpy.ndarray
            An array containing predicted classes numbers.

        """
        series_class_votes = np.zeros(
            (len(series_list), len(self.class_models)), dtype=int
        )
        series_class_weights = np.zeros((len(series_list), len(self.class_models)))

        for class_model in self.class_models:
            membership_list = cmeans.find_memberships(series_list, class_model[1])
            current_class_binary_classifiers = list(
                filter(
                    lambda x, class_number=class_model[0]: x.class_numbers[0]
                    == class_number,
                    self._binary_classifiers,
                )
            )
            for series_idx, membership_matrix in enumerate(membership_list):
                for binary_classifier in current_class_binary_classifiers:
                    predicted_class, output_weights = binary_classifier.predict(
                        membership_matrix
                    )
                    series_class_votes[series_idx, predicted_class - 1] += 1
                    series_class_weights[
                        series_idx, np.subtract(binary_classifier.class_numbers, 1)
                    ] += output_weights

        return self._prepare_result(series_class_votes, series_class_weights)

    @staticmethod
    def _prepare_result(series_class_votes, series_class_weights):
        result = np.argmax(series_class_votes, axis=1)

        max_entries_in_row = (
            series_class_votes == series_class_votes.max(axis=1)[:, np.newaxis]
        )
        max_entries_count_per_row = max_entries_in_row.sum(axis=1)
        ambiguous_rows = max_entries_count_per_row > 1
        # The next line is necessary to properly handle situations such that class with less votes
        # in total has greater sum of weights than classes with max number of votes.
        series_class_weights[np.logical_not(max_entries_in_row)] = -1

        result[ambiguous_rows] = np.argmax(series_class_weights[ambiguous_rows], axis=1)
        result = result + 1

        return result

    def save(self, filename):
        """
        Save file.

        Parameters
        ----------
        filename : string

        Returns
        -------
        None.

        """
        with open(filename, "wb") as file:
            pickle.dump(self, file, protocol=min(4, pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, filename):
        """
        Load file.

        Parameters
        ----------
        filename : string

        Returns
        -------
        object

        """
        with open(filename, "rb") as file:
            return pickle.load(file)
