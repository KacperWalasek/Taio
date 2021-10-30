"""
This module contains create_fcms function which should be used to create
fuzzy cognitive map(s) for given series. Under the hood, genetic
algorithm is used.
"""
import functools
from itertools import accumulate, starmap
from sys import platform
import numpy as np
import pygad
import skfuzzy as fuzz

if platform in ["linux", "linux2"]:
    from ray.util.multiprocessing import Pool
else:
    from multiprocessing import Pool


def _sigmoid(x):
    return 1 / (1 + np.exp(-5 * x))


def _g_func(x):
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
    tuple
        The first element of returned tuple is numpy.ndarray containing
        cluster centers. The second one is a numpy.ndarray of shape (N, c)
        in which [i, j]th element is membership
        function's value of i-th point to j-th concept (centroid).

    """
    m = 2
    error = 1e-8
    maxiter = 1e8
    ret = fuzz.cmeans(coordinates, concept_count, m, error, maxiter)
    return (
        ret[0],
        ret[1].T,
    )


def _cmeans_predict_wrapper(coordinates, cluster_centers):
    m = 2
    error = 1e-8
    maxiter = 1e8
    return fuzz.cmeans_predict(coordinates, cluster_centers, m, error, maxiter)[0].T


def _fuzzify_training_series_list(series_list, concept_count, split):
    combined_coordinates_list = []
    for series in series_list:
        adjacent_differences = np.diff(series)
        series = series[:, 1:]
        combined_coordinates = np.hstack((series, adjacent_differences)).reshape(
            -1, series.shape[1]
        )
        combined_coordinates_list.append(combined_coordinates)
    horizontal_combined_coordinates = np.hstack(combined_coordinates_list)
    if split:
        horizontal_coordinates_list = np.vsplit(
            horizontal_combined_coordinates, horizontal_combined_coordinates.shape[0]
        )
    else:
        horizontal_coordinates_list = [horizontal_combined_coordinates]
    # W tym momencie horizontal_coordinates_list to lista rzeczy do obrobienia c_meansami
    fun = functools.partial(_cmeans_wrapper, concept_count=concept_count)
    cluster_centers_list, membership_matrices = zip(
        *map(fun, horizontal_coordinates_list)
    )
    # membership_matrices_per_series będzie listą o długości len(series_list)
    # taką, że na indeksie i-tym
    # będą odpowiadające macierze przynależności dla i-tego szeregu
    # membership_matrices_per_series[i] jest listą o długości albo 3 albo 1
    # (zależnie od parametru split) i zawiera numpyowe arraye przynależności
    membership_matrices_per_series = list(
        map(
            list,
            zip(
                *map(
                    functools.partial(
                        np.vsplit,
                        indices_or_sections=[
                            x - 1
                            for x in accumulate([y.shape[1] for y in series_list[:-1]])
                        ],
                    ),
                    membership_matrices,
                )
            ),
        )
    )
    # cluster_centers_list jest lista o długości zależnej od split
    return cluster_centers_list, membership_matrices_per_series


def _fuzzify_test_series_list(series_list, cluster_centers_list, split):
    combined_coordinates_list = []
    for series in series_list:
        adjacent_differences = np.diff(series)
        series = series[:, 1:]
        combined_coordinates = np.hstack((series, adjacent_differences)).reshape(
            -1, series.shape[1]
        )
        combined_coordinates_list.append(combined_coordinates)
    horizontal_combined_coordinates = np.hstack(combined_coordinates_list)
    if split:
        horizontal_coordinates_list = np.vsplit(
            horizontal_combined_coordinates, horizontal_combined_coordinates.shape[0]
        )
    else:
        horizontal_coordinates_list = [horizontal_combined_coordinates]
    membership_matrices = starmap(
        _cmeans_predict_wrapper, zip(horizontal_coordinates_list, cluster_centers_list)
    )
    membership_matrices_per_series = list(
        map(
            list,
            zip(
                *map(
                    functools.partial(
                        np.vsplit,
                        indices_or_sections=[
                            x - 1
                            for x in accumulate([y.shape[1] for y in series_list[:-1]])
                        ],
                    ),
                    membership_matrices,
                )
            ),
        )
    )
    return membership_matrices_per_series


def _create_single_series_fcms(
    membership_matrices, previous_considered_indices, concept_count
):
    trained_array_size = (
        previous_considered_indices.size * concept_count + concept_count ** 2
    )
    fcm_list = []
    for membership_matrix in membership_matrices:
        fitness_func = _create_fitness_func(
            membership_matrix, previous_considered_indices, concept_count
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


def create_training_fcms(
    series_list, previous_considered_indices, concept_count, split
):
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
    tuple
        The first element of tuple is a list of 2d numpy.ndarrays which
        are coordinates of cluster centers. This list can be later
        used as an argument of create_test_fcms function. If split==False
        then this list contains only one cluster centers ndarray.
        The second element of tuple is a list of lists containing FCMs
        (or one FCM) of type numpy.ndarray. Each element of the outer list
        corresponds to one series in series_list.

    """
    cluster_centers_list, membership_matrices_list = _fuzzify_training_series_list(
        series_list, concept_count, split
    )
    fun = functools.partial(
        _create_single_series_fcms,
        previous_considered_indices=previous_considered_indices,
        concept_count=concept_count,
    )
    with Pool() as p:
        fcms_list = p.map(fun, membership_matrices_list)

    return cluster_centers_list, fcms_list


def create_test_fcms(
    series_list, previous_considered_indices, concept_count, split, cluster_centers
):
    """
    TU COS WPISAC
    """
    membership_matrices_list = _fuzzify_test_series_list(
        series_list, cluster_centers, split
    )
    fun = functools.partial(
        _create_single_series_fcms,
        previous_considered_indices=previous_considered_indices,
        concept_count=concept_count,
    )
    with Pool() as p:
        fcms_list = p.map(fun, membership_matrices_list)

    return fcms_list


if __name__ == "__main__":
    example_series = np.ones((3, 100))
    example_series_2 = -2 * np.ones((3, 200))
    example_previous_indices = np.array([1, 2, 3])
    cluster_centers_result, fcms_result = create_training_fcms(
        [example_series, example_series_2], example_previous_indices, 2, True
    )
    fcms_test_result = create_test_fcms(
        [example_series, example_series_2],
        example_previous_indices,
        2,
        True,
        cluster_centers_result,
    )
