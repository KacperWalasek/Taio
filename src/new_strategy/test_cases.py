"""
Module with test cases.
"""
import os
import series_classification


class TestCase:
    """
    TestCase class.
    """

    def __init__(self, length_percent, previous_considered_indices, move):
        self.length_percent = length_percent
        self.previous_considered_indices = previous_considered_indices
        self.move = move
        self.result = 0

    def run(self):
        """
        Run train and test functions.
        """
        models, class_count = series_classification.train(
            os.path.join("UWaveGestureLibrary_Preprocessed", "Train"),
            self.length_percent,
            self.previous_considered_indices,
            self.move,
        )
        self.result = series_classification.test(
            os.path.join("UWaveGestureLibrary_Preprocessed", "Test"),
            self.length_percent,
            self.previous_considered_indices,
            self.move,
            models,
            class_count,
        )

    def get_result(self):
        """
        Get result of test case.
        """
        return self.result


if __name__ == "__main__":
    tests = []
    tests.append(TestCase(1, [1, 2], 1))

    for test in tests:
        test.run()
        print("Test result: ", test.get_result())
