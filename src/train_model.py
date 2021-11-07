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
        classes_memberships,
        clusters,
        length_percent,
        previous_considered_indices,
        move,
    ):
        multiprocessing.Process.__init__(self)
        self.class1 = classes_memberships[0][0]
        self.class2 = classes_memberships[1][0]
        self.memberships1 = classes_memberships[0][1]
        self.memberships2 = classes_memberships[1][1]
        self.clusters = clusters
        self.length_percent = length_percent
        self.previous_considered_indices = previous_considered_indices
        self.move = move
        self.matrices = []

    def run(self):
        # tu trzeba cos napisac
        self.matrices = [0]
