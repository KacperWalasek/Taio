"""
Module with tests.
"""
import os
import series_classification


class Test:
    """
    Test class.
    """

    def __init__(self, length_percent, previous_considered_indices, split):
        self.percent = length_percent
        self.prev = previous_considered_indices
        self.split = split
        self.result = 0

    def run(self):
        """
        Run train and test functions.
        """
        series_classification.train(
            os.path.join("UWaveGestureLibrary_Preprocessed", "Train"),
            self.percent,
            self.prev,
            self.split,
        )
        self.result = series_classification.test(
            os.path.join("UWaveGestureLibrary_Preprocessed", "Test"),
            self.percent,
            self.prev,
            self.split,
        )

    def get_result(self):
        """
        Get result of test case.
        """
        return self.result


if __name__ == "__main__":
    tests = []
    tests.append(Test(1, [1, 2], True))

    for test in tests:
        test.run()
