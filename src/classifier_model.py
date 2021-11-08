"""
Classifier model (2 classification classes).
"""
import numpy as np
import pygad


class ClassifierModel:
    """
    Train one model with run function.

    Parameters
    ----------
    class_numbers : tuple
        Tuple containing two class numbers.
    membership_matrices : tuple
        Tuple containing two lists of series membership matrices.
        Both elements of tuple correspond to classes specified in class_numbers.
    previous_considered_indices: numpy.ndarray
        Array containing indices of previous elements which will be
        FCM's input for predicting the next one.
    move: int
        Step of frame used in processing single series.

    """

    _GA_PARAMS = {
        "num_generations": 1000,
        "sol_per_pop": 20,
        "num_parents_mating": 10,
        "gene_space": {"low": -1, "high": 1},
        "gene_type": np.float64,
        "mutation_type": "random",
    }

    def __init__(
        self,
        class_numbers,
        membership_matrices,
        previous_considered_indices,
        move,
    ):

        self.class_numbers = class_numbers
        self._membership_matrices = membership_matrices
        self._previous_considered_indices = previous_considered_indices
        self._move = move

        self._concept_count = membership_matrices[0][0].shape[1]
        self._uwv_matrices = []
        self.is_trained = False

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-5 * x))

    def _classify_series(self, membership_matrix, u_matrix, w_matrix, v_matrix):

        max_previous_index = self._previous_considered_indices.max()

        # Note that targets for which we calculate predictions are
        # membership_matrix[max_previous_index:membership_matrix.shape[0]].
        # For every predicion we create an array containing indices
        # of its input elements.
        input_indices = (
            np.arange(max_previous_index, membership_matrix.shape[0], self._move)[
                :, np.newaxis
            ].repeat(self._previous_considered_indices.size, axis=1)
            - self._previous_considered_indices
        )

        # Now we compute matrix containing a-values for each target.
        # The number of columns is equal to concept_count (each target element
        # corresponds to single row which contains a-value for each concept).
        a_matrix = ClassifierModel._sigmoid(
            (membership_matrix[input_indices] * u_matrix).sum(axis=1)
        )

        # We calculate predicted values for each target.
        predicted_concept_membership_matrix = ClassifierModel._sigmoid(
            np.matmul(a_matrix, w_matrix.T)
        )

        # The last step - class prediction.
        # v_matrix must be of shape (2, concept_count)
        predicted_class_memberships = ClassifierModel._sigmoid(
            (predicted_concept_membership_matrix[:, np.newaxis] * v_matrix).sum(axis=2)
        )
        discrete_class_memberships = np.zeros_like(
            predicted_class_memberships, dtype=int
        )
        discrete_class_memberships[:, predicted_class_memberships.argmax(1)] = 1
        total_votes = discrete_class_memberships.sum(axis=0)
        total_weights = predicted_class_memberships.sum(axis=0)
        chosen_class_idx = (
            total_votes.argmax()
            if total_votes[0] != total_votes[1]
            else total_weights.argmax()
        )
        return self.class_numbers[chosen_class_idx], total_weights[chosen_class_idx]

    def _split_uwv_array(self, array):
        w_matrix_offset = self._previous_considered_indices.size * self._concept_count
        v_matrix_offset = w_matrix_offset + self._concept_count ** 2
        u_matrix = array[:w_matrix_offset].reshape(-1, self._concept_count)
        w_matrix = array[w_matrix_offset:v_matrix_offset].reshape(
            -1, self._concept_count
        )
        v_matrix = array[v_matrix_offset:].reshape(2, -1)
        return u_matrix, w_matrix, v_matrix

    def _create_fitness_func(self):
        def fitness_func(solution, _):
            u_matrix, w_matrix, v_matrix = self._split_uwv_array(solution)
            misclassified_count = 0
            for idx, class_number in enumerate(self.class_numbers):
                for membership_matrix in self._membership_matrices[idx]:
                    assigned_class_number = self._classify_series(
                        membership_matrix, u_matrix, w_matrix, v_matrix
                    )[0]
                    if assigned_class_number != class_number:
                        misclassified_count += 1
            return -misclassified_count

        return fitness_func

    def train(self):
        """
        Method for training classifier instance.

        Returns
        -------
        None.

        """
        fitness_func = self._create_fitness_func()
        trained_array_size = (
            self._previous_considered_indices.size * self._concept_count
            + self._concept_count ** 2
            + 2 * self._concept_count
        )
        ga_instance = pygad.GA(
            num_genes=trained_array_size,
            fitness_func=fitness_func,
            **self._GA_PARAMS,
        )
        ga_instance.run()
        # ga_instance.plot_fitness()
        solution, solution_fitness, _ = ga_instance.best_solution()
        print(f"Best solution fitness (SSE): {-solution_fitness}")

        self._uwv_matrices = self._split_uwv_array(solution)
        self.is_trained = True

    def classify_series(self, membership_matrix):
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
        return self._classify_series(membership_matrix, *self._uwv_matrices)


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
    model = ClassifierModel((5, 6), test_membership_matrices, test_previous_indices, 2)
    model.train()
    if (
        model.classify_series(test_membership_1)[0] == 5
        and model.classify_series(test_membership_2)[0] == 5
        and model.classify_series(test_membership_3)[0] == 6
    ):
        print("OK")
