"""
Module with test cases.
"""

import sys
import os
import series_classification


class TestCase:  # pylint: disable=too-few-public-methods
    """
    TestCase class.

    Parameters
    ----------
    length_percent : list
        list of numbers from range [0, 1]
    previous_considered_indices : list
        indicies specifying window, e.g. [1, 2, 3] means that the window
        consists of three preceding (consecutive) elements
    move : int
        window move step
    concept_count : int
        number of centroids for fuzzy c-means clustering
    """

    def __init__(
        self, length_percentages, previous_considered_indices, move, concept_count
    ):
        self._length_percentages = length_percentages
        self._previous_considered_indices = previous_considered_indices
        self._move = move
        self._concept_count = concept_count

    def run(self, dataset_dir, run_id):
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
            os.path.join(dataset_dir, "Train"),
            self._previous_considered_indices,
            self._move,
            self._concept_count,
        )
        name = os.path.basename(os.path.normpath(dataset_dir))
        series_classifier.save(f"{name}_model{run_id}.dat")
        for length_percent in self._length_percentages:
            train_result = series_classification.test(
                os.path.join(dataset_dir, "Train"),
                length_percent,
                series_classifier,
            )
            self._print_result(train_result, f"{name} Train", length_percent, run_id)
            test_result = series_classification.test(
                os.path.join(dataset_dir, "Test"),
                length_percent,
                series_classifier,
            )
            self._print_result(test_result, f"{name} Test", length_percent, run_id)

    def _print_result(self, result, series_info, length_percent, run_id):
        print("-" * 20)
        print(
            f"{series_info} result: {result}",
            f"Length percent: {length_percent}",
            f"Previous considered indices: {self._previous_considered_indices}",
            f"Move: {self._move}",
            f"Concept count: {self._concept_count}",
            f"Run id: {run_id}",
            sep="\n",
            flush=True,
        )
        print("-" * 20)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError("Too few arguments, dataset directory path is missing")
    data_dir = sys.argv[1]

    tests = []
    tests.append(TestCase([1, 0.8], [1, 2, 3, 4, 5, 6], 1, 15))

    for test_id, test in enumerate(tests):
        test.run(data_dir, test_id)
