"""
Class for one class model - clusters.
"""
import os
import multiprocessing
import read_data


class ClassModel(multiprocessing.Process):
    """
    Get class clusters with run function.
    """

    def __init__(
        self, class_number, path, length_percent, previous_considered_indices, move
    ):
        multiprocessing.Process.__init__(self)

        self.class_number = class_number
        self.path = path
        self.length_percent = length_percent
        self.previous_considered_indices = previous_considered_indices
        self.move = move
        self.series_list = []
        self.clusters = []

    def get_series_list(self):
        """
        Get list of time series in directory (self.path).
        """
        series_list = []
        for file in os.scandir(self.path):
            if file.name.endswith(".csv"):
                series_list.append(
                    read_data.process_data(file.path, self.length_percent)
                )

        # tu lista jest spoko i działa
        self.series_list = multiprocessing.Manager().list(series_list)

    def run(self):
        self.get_series_list()

        self.clusters = [0]  # tu maja sie liczyc klastry
