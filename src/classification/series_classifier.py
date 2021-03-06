"""
Class for classifying list of time series.
"""

import pickle
import numpy as np
import preprocessing.cmeans_clustering as cmeans
import params
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
        self.class_models = list(map(lambda x: x[0], class_models))
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
            (len(series_list), len(self.class_models)), dtype=np.int32
        )
        series_class_weights = np.zeros((len(series_list), len(self.class_models)))

        for class_idx in range(len(self.class_models)):
            current_class_binary_classifiers = list(
                filter(
                    lambda x, class_idx=class_idx: x.class_numbers[0] == class_idx,
                    self._binary_classifiers,
                )
            )
            for binary_classifier in current_class_binary_classifiers:
                membership_list = cmeans.find_memberships(series_list, binary_classifier.centroids)
                for series_idx, membership_matrix in enumerate(membership_list):
                    predicted_class_idx, output_weights = binary_classifier.predict(
                        membership_matrix
                    )
                    if predicted_class_idx != -1:
                        series_class_votes[series_idx, predicted_class_idx] += 1
                    if binary_classifier.class_numbers[1] == -1:
                        series_class_weights[
                            series_idx,binary_classifier.class_numbers[0]
                        ] += output_weights[0]
                    else:
                        series_class_weights[
                            series_idx, binary_classifier.class_numbers
                        ] += output_weights


        result_indices = self._prepare_result_indices(
            series_class_votes, series_class_weights
        )
        return np.array([x for x in self.class_models])[result_indices]

    @staticmethod
    def _prepare_result_indices(series_class_votes, series_class_weights):
        result_indices = np.argmax(series_class_votes, axis=1)

        max_entries_in_row = (
            series_class_votes == series_class_votes.max(axis=1)[:, np.newaxis]
        )
        max_entries_count_per_row = max_entries_in_row.sum(axis=1)
        ambiguous_rows = max_entries_count_per_row > 1
        
        # The next line is necessary to properly handle situations such that class with less votes
        # in total has greater sum of weights than classes with max number of votes.
        series_class_weights[np.logical_not(max_entries_in_row)] = -1

        result_indices[ambiguous_rows] = np.argmax(
            series_class_weights[ambiguous_rows], axis=1
        )

        return result_indices

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
