"""
Class for one class model - clusters.
"""
import os

import read_data
import cmeans_clustering

class ClassModel():
    """
    Get class clusters with run function.
    """

    def __init__(
        self, class_number, path, length_percent
    ):
        self.class_number = class_number
        self.path = path
        self.length_percent = length_percent
        self.series_list = []
        self.centroids = []
        self.memberships = []

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
        self.series_list = series_list #multiprocessing.Manager().list(series_list)

    def run(self):
        print(self.class_number)
        self.get_series_list()
        self.centroids, self.memberships = cmeans_clustering.find_clusters(self.series_list, 12)
        