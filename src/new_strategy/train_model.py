"""
Class for one train model (2 classification classes).
"""
import multiprocessing


class TrainModel(multiprocessing.Process):
    """
    Train one model with run function.
    """

    def __init__(
        self,
        classes_series_lists,
        clusters,
        length_percent,
        previous_considered_indices,
        move,
    ):
        multiprocessing.Process.__init__(self)

        self.class1 = classes_series_lists[0][0]
        self.class2 = classes_series_lists[1][0]
        self.series_list1 = classes_series_lists[0][1]
        self.series_list2 = classes_series_lists[1][1]
        self.clusters = clusters
        self.length_percent = length_percent
        self.previous_considered_indices = previous_considered_indices
        self.move = move
        self.matrices = []

    def run(self):
        # tu trzeba cos napisac
        self.matrices = [0]
