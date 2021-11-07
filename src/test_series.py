"""
Class for testing one time series.
"""
import multiprocessing


class Test(multiprocessing.Process):
    """
    Test one time series with run function.
    """

    def __init__(
        self,
        class_number,
        time_series,
        previous_considered_indices,
        move,
        models,
        class_count,
    ):
        multiprocessing.Process.__init__(self)

        self.class_number = class_number
        self.time_series = time_series
        self.previous_considered_indices = previous_considered_indices
        self.move = move
        self.models = models
        self.class_count = class_count
        self.result = multiprocessing.Value("b", False)

    def run(self):
        classes = [0 for _ in range(self.class_count)]

        for model in self.models:
            result = 1  # tu trzeba cos napisac
            classes[result - 1] = classes[result - 1] + 1

        predicted_class = classes.index(max(classes)) + 1
        self.result.value = predicted_class == self.class_number
