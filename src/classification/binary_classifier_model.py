"""
BinaryClassifierModel model (2 classification classes).
"""

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import AlgorithmParams
from params import _GA_PARAMS, _GA_RUN_PARAMS
import functools
import classification.computing_utils as computing_utils


class BinaryClassifierModel:
    """
    Class representing single class vs class classifier model.

    Parameters
    ----------
    class_numbers : tuple
        Tuple containing two class numbers.
    membership_matrices : tuple
        Tuple containing two lists of series membership matrices.
        Both elements of tuple correspond to classes specified in class_numbers.
    centroids: numpy.ndarray
        Ndarray of shape (concept_count, centroids_space_dimension) containing
        coordinates of centroids.
    previous_considered_indices: list
        List containing indices of previous elements which will be
        FCM's input for predicting the next one.
    move: int
        Step of frame used in processing single series.

    """

    def __init__(
        self,
        class_numbers,
        membership_matrices,
        centroids,
        previous_considered_indices,
        move,
    ):

        self.class_numbers = class_numbers
        self._membership_matrices = membership_matrices
        self.centroids = centroids
        self._previous_considered_indices = np.array(
            previous_considered_indices, dtype=np.int32
        )
        self._move = move

        self._concept_count = membership_matrices[0][0].shape[1]
        self._uwv_matrices = []
        self.is_trained = False

    def _create_fitness_func(self):
        return functools.partial(computing_utils.fitness_func, membership_matrices = self._membership_matrices, 
                previous_considered_indices = self._previous_considered_indices,
                move = self._move)

    def train(self):
        """
        Method for training classifier instance.

        Returns
        -------
        None.

        """
        if self.is_trained:
            return self
        fitness_func = self._create_fitness_func()
        trained_array_size = (
            self._previous_considered_indices.size * self._concept_count
            + self._concept_count ** 2
            + 2 * self._concept_count
        )
        bounds = np.array([[-1, 1]] * trained_array_size)
        ga_model = ga(
            function = fitness_func,
            dimension=trained_array_size,
            variable_boundaries=bounds,
            **_GA_PARAMS,
        )
        ga_model.run(
            set_function=ga.set_function_multiprocess(fitness_func),
            stop_when_reached=0,
            **_GA_RUN_PARAMS,
        )

        solution = ga_model.output_dict["variable"]
        print(
            f"Fraction of misclassified series for classifier {self.class_numbers}: "
            f"{ga_model.output_dict['function']/sum((len(x) for x in self._membership_matrices))}",
            flush=True,
        )

        self._uwv_matrices = computing_utils.split_uwv_array(solution, self._previous_considered_indices.size, self._concept_count)
        self.is_trained = True
        del self._membership_matrices
        return self

    def predict(self, membership_matrix):
        """
        Method which classifies series to one of the classes specified
        in self.class_numbers tuple.

        Parameters
        ----------
        membership_matrix : numpy.ndarray
            Membership matrix of input series.

        Raises
        ------
        RuntimeError
            Classifier has not been trained.

        Returns
        -------
        tuple
            The first element is calculated class number and the second
            one is the sum of model's output weights for this class.

        """
        if not self.is_trained:
            raise RuntimeError("Classifier has not been trained")
        # pylint: disable = no-value-for-parameter
        prediction = computing_utils.predict_series_class_idx(membership_matrix, *self._uwv_matrices, self._previous_considered_indices, self._move)
        return self.class_numbers[prediction[0]], prediction[1]


if __name__ == "__main__":
    test_membership_1 = np.tile(
        np.array([[1, 0, 0], [0.5, 0.5, 0], [0.25, 0.5, 0.25]]), (30, 1)
    )
    test_membership_2 = np.tile(
        np.array([[0.75, 0.2, 0.15], [0.5, 0.5, 0], [0.2, 0.5, 0.3], [0.6, 0.2, 0.2]]),
        (40, 1),
    )
    test_membership_3 = np.tile(np.array([[0.3, 0.4, 0.3], [0.7, 0.2, 0.1]]), (30, 1))
    test_membership_matrices = (
        [test_membership_1, test_membership_2],
        [test_membership_3],
    )
    test_previous_indices = np.r_[1:4]
    model = BinaryClassifierModel(
        (5, 6), test_membership_matrices, None, test_previous_indices, 2
    )

    model.train()
    if (
        model.predict(test_membership_1)[0] == 5
        and model.predict(test_membership_2)[0] == 5
        and model.predict(test_membership_3)[0] == 6
    ):
        print("OK")
    else:
        print("NOT OK")
