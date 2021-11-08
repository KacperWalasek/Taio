"""
Class for classifying list of time series.
"""

import numpy as np
import cmeans_clustering as cmeans


class SeriesClassifier:  # pylint: disable=too-few-public-methods
    """
    A class for classifying time series based upon trained binary classifiers.

    Parameters
    ----------
    class_models : list
        A list of objects of type ClassModel.
    binary_classifiers : list
        A list of previously trained BinaryClassifier objects corresponding to class_models.

    """

    def __init__(self, class_models, binary_classifiers):
        self.class_models = class_models
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
            membership_list = cmeans.find_memberships(
                series_list, class_model.centroids
            )
            current_class_binary_classifiers = filter(
                lambda x, class_number=class_model.class_number: x.class_numbers[0]
                == class_number,
                self._binary_classifiers,
            )
            for series_idx, membership_matrix in enumerate(membership_list):
                for binary_classifier in current_class_binary_classifiers:
                    predicted_class, predicted_class_weight = binary_classifier.predict(
                        membership_matrix
                    )
                    col_idx = predicted_class - 1
                    series_class_votes[series_idx, col_idx] += 1
                    # pylint: disable=fixme
                    # TODO: Sprawdź też przydzielanie wag w tym momencie również dla klasy gorszej
                    # - chyba to będzie lepsze podejście
                    series_class_weights[series_idx, col_idx] += predicted_class_weight

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
