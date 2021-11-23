"""
Module with test cases.
"""

import os
import series_classification


class TestCase:
    """
    TestCase class.

    Parameters
    ----------
    length_percent : number
        [0, 1]
    previous_considered_indices : list
    move : int
    concept_count : int

    """

    def __init__(
        self, length_percent, previous_considered_indices, move, concept_count=12
    ):
        self._length_percent = length_percent
        self._previous_considered_indices = previous_considered_indices
        self._move = move
        self._train_result = 0
        self._test_result = 0
        self._concept_count = concept_count

    def run(self):
        """
        Run train and test functions.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        series_classifier = series_classification.train(
            os.path.join("UWaveGestureLibrary_Preprocessed", "Train"),
            self._length_percent,
            self._previous_considered_indices,
            self._move,
            self._concept_count,
        )
        self._train_result = series_classification.test(
            os.path.join("UWaveGestureLibrary_Preprocessed", "Train"),
            self._length_percent,
            series_classifier,
        )
        self._test_result = series_classification.test(
            os.path.join("UWaveGestureLibrary_Preprocessed", "Test"),
            self._length_percent,
            series_classifier,
        )

    def get_train_result(self):
        """
        Get result of train test case.

        Parameters
        ----------
        None.

        Returns
        -------
        number

        """
        return self._train_result

    def get_test_result(self):
        """
        Get result of test test case.

        Parameters
        ----------
        None.

        Returns
        -------
        number

        """
        return self._test_result


if __name__ == "__main__":
    tests = []
    tests.append(TestCase(1, [1, 2, 3, 4, 5, 6], 1, 15))

    for test in tests:
        test.run()
        print("Train result: ", test.get_train_result())
        print("Test result: ", test.get_test_result())
