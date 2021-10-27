"""
This module contains create_fcms function which should be used to create
fuzzy cognitive map(s) for given series. Under the hood, genetic
algorithm is used.
"""
import functools
import numpy as np
import pygad
import skfuzzy as fuzz


def _sigmoid(x):
    """
    Sigmoid function with parameter 5 compatible with numpy.ndarray.
    """
    return 1 / (1 + np.exp(-5 * x))


def _g_func(x):
    """
    Internal function used to modify the outputs of first training layer
    (the one with u weights).
    """
    return _sigmoid(x)


def _create_fitness_func(membership_matrix, previous_considered_indices, concept_count):
    """
    Creates fitness function for genetic algorithm.

    Parameters
    ----------
    membership_matrix : numpy.ndarray
        Membership matrix of series for which we want to create FCM, generated
        with fuzzy c-means clustering algorithm.
    previous_considered_indices : numpy.ndarray
        Array containing indices of previous elements which will be
        FCM's input for predicting the next one. For example, if you want
        to predict next element using the current one, the previous one and
        before the previous one you can pass numpy.array([1, 2, 3]).
        This argument's entries do not have to be sorted.
    concept_count : int
        Number of concepts. This argument should be equal
        to membership_matrix.shape[1].

    Returns
    -------
    function
        Fitness function ready to be used in genetic algorithm. Its first
        parameter is a single dimensional numpy.ndarray which contains two
        flattened matrices, the first one with u weights and the second one
        with w weights (FCM). The offset for w weights is equal to
        previous_considered_indices.size * concept_count.
        The second argument is obligatory, corresponds to solution index
        and in this case is unused.

    """
    w_matrix_offset = previous_considered_indices.size * concept_count
    max_previous_index = previous_considered_indices.max()

    def fitness_func(solution, _):
        u_matrix = solution[:w_matrix_offset].reshape(-1, concept_count)
        w_matrix = solution[w_matrix_offset:].reshape(-1, concept_count)

        # Note that targets for which we calculate predictions are
        # membership_matrix[max_previous_index:membership_matrix.shape[0]].
        # For every predicion we create an array containing indices
        # of its input elements.
        input_indices = (
            np.arange(max_previous_index, membership_matrix.shape[0])[
                :, np.newaxis
            ].repeat(previous_considered_indices.size, axis=1)
            - previous_considered_indices
        )

        # Now we compute matrix containing a-values for each target.
        # The number of columns is equal to concept_count (each target element
        # corresponds to single row which contains a-value for each concept).
        a_matrix = _g_func((membership_matrix[input_indices] * u_matrix).sum(axis=1))

        # We calculate predicted values for each target...
        y_matrix = _sigmoid(np.matmul(a_matrix, w_matrix.T))

        # and the original ones.
        target_matrix = membership_matrix[max_previous_index:]

        # The last step - error calculation.
        sse = ((y_matrix - target_matrix) ** 2).sum()

        # Note that PyGAD maximizes fitness function.
        return -sse

    return fitness_func


def _cmeans_wrapper(coordinates, concept_count):
    """
    Function which wraps fuzz.cmeans.
    Read more at https://scikit-fuzzy.github.io/scikit-fuzzy/api/index.html

    Parameters
    ----------
    coordinates : numpy.ndarray
        An array of shape (space_dim, N) of points to cluster.
        N is the number of points, space_dim is the dimension of space
        to which given points belong.
    concept_count : int
        Number of concepts (centroids).

    Returns
    -------
    numpy.ndarray
        An array of shape (N, c) in which [i, j]th element is membership
        function's value of i-th point to j-th concept.

    """
    m = 2
    error = 1e-8
    maxiter = 1e8
    return fuzz.cmeans(coordinates, concept_count, m, error, maxiter)[1].T


def _fuzzify_series(series, concept_count, split):
    """
    Internal function used to create membership matrix for each coordinate
    of series. The series elements are clustered using fuzzy c-means clustering.

    Parameters
    ----------
    series : numpy.ndarray
        Input series, does not have to be normalized.
        Note that its shape must be (space_dim, N) where N is a number of
        observations in the series and space_dim is the dimension of space
        to which given points belong.
    concept_count : int
        Number of concepts (centroids).
    split : bool
        If True, we get membership matrices for each coordinate of series.
        If False, we get one membership matrix for all coordinates together.

    Returns
    -------
    list
        A list containing membership matrices (for each coordinate of series
        or one for all coordinates together).
        The matrices are of type numpy.ndarray and of shape (N, concept_count).

    """
    adjacent_differences = np.diff(series)
    series = series[:, 1:]
    combined_coordinates = np.hstack((series, adjacent_differences)).reshape(
        -1, series.shape[1]
    )
    if split:
        coordinates_list = np.vsplit(combined_coordinates, series.shape[0])
    else:
        coordinates_list = [combined_coordinates]
    fun = functools.partial(_cmeans_wrapper, concept_count=concept_count)
    membership_matrices = list(map(fun, coordinates_list))
    return membership_matrices


def create_fcms(series, previous_considered_indices, concept_count, split):
    """
    Creates FCMs for given multidimensional series - one FCM for one coordinate.

    Parameters
    ----------
    series : numpy.ndarray
        Input series, does not have to be normalized.
        Note that its shape must be (space_dim, N) where N is a number of
        observations in the series and space_dim is the dimension of space
        to which given points belong.
    previous_considered_indices : numpy.ndarray
        Array containing indices of previous elements which will be
        FCM's input for predicting the next one. For example, if you want
        to predict next element using the current one, the previous one and
        before the previous one you can pass numpy.array([1, 2, 3]).
        This argument's entries do not have to be sorted.
    concept_count : int
        Number of concepts (centroids).
    split : bool
        If True, we get series.shape[0] number of FCMs.
        If False, we get one FCM.

    Returns
    -------
    list
        List containing FCMs (or one FCM) of type numpy.ndarray.

    """
    membership_matrices = _fuzzify_series(series, concept_count, split)
    trained_array_size = (
        previous_considered_indices.size * concept_count + concept_count ** 2
    )
    fcm_list = []
    for matrix in membership_matrices:
        fitness_func = _create_fitness_func(
            matrix, previous_considered_indices, concept_count
        )

        ga_instance = pygad.GA(
            num_generations=5,
            sol_per_pop=20,
            num_parents_mating=10,
            num_genes=trained_array_size,
            gene_space={"low": -1, "high": 1},
            gene_type=np.float64,
            fitness_func=fitness_func,
            mutation_type="random",
        )
        ga_instance.run()
        # ga_instance.plot_fitness()

        solution, solution_fitness, _ = ga_instance.best_solution()
        print(f"Best solution fitness (SSE): {-solution_fitness}")

        w_matrix_offset = previous_considered_indices.size * concept_count
        fcm_list.append(solution[w_matrix_offset:].reshape(concept_count, -1))

    return fcm_list


if __name__ == "__main__":
    example_series = np.array([0, 0.5, 1] * 100)[:, np.newaxis].repeat(3, axis=1)
    example_series[:, 1] = example_series[:, 1] / 3
    example_series[:, 2] = ((example_series[:, 2] + 0.5) % 1) ** 2
    example_previous_indices = np.r_[1:7]
    example_series = example_series.T
    fcms = create_fcms(example_series, example_previous_indices, 2, True)
