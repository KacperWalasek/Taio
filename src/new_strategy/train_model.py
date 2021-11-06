"""
Class for one train model (2 classification classes).
"""
import os
import multiprocessing
import read_data


class TrainModel(multiprocessing.Process):
    """
    Train one model with run function.
    """

    def __init__(
        self, classes_paths, length_percent, previous_considered_indices, move
    ):
        multiprocessing.Process.__init__(self)

        self.class1 = classes_paths[0][0]
        self.class2 = classes_paths[1][0]
        self.path1 = classes_paths[0][1]
        self.path2 = classes_paths[1][1]
        self.length_percent = length_percent
        self.previous_considered_indices = previous_considered_indices
        self.move = move
        self.matrices = 0

    def get_series_list(self, path):
        """
        Get list of time series in directory given (path).
        """
        series_list = []
        for file in os.scandir(path):
            if file.name.endswith(".csv"):
                series_list.append(
                    read_data.process_data(file.path, self.length_percent)
                )
        return series_list

    def run(self):
        series_list1 = self.get_series_list(self.path1)
        series_list2 = self.get_series_list(self.path2)
        # tu trzeba cos napisac
        self.matrices = 0
