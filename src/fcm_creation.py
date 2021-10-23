"""
This module contains create_fcm function which should be used to create
fuzzy cognitive map for given series. Under the hood, genetic
algorithm is used.
"""
import numpy as np
import pygad


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


def _create_fitness_func(series, previous_considered_indices, space_dim):
    """
    Creates fitness function for genetic algorithm.

    Parameters
    ----------
    series : numpy.ndarray
        Series for which we want to create FCM. Important note: the series
        must be normalized to [0, 1] interval (all coordinates must
        belong to it). The argument must be two dimensional.
    previous_considered_indices : numpy.ndarray
        Array containing indices of previous elements which will be
        FCM's input for predicting the next one. For example, if you want
        to predict next element using the current one, the previous one and
        before the previous one you can pass numpy.array([1, 2, 3]).
        This argument's entries do not have to be sorted.
    space_dim : int
        Dimension of space of series' entries. This argument should be equal
        to series.shape[1].

    Returns
    -------
    function
        Fitness function ready to be used in genetic algorithm. Its first
        parameter is a single dimensional numpy.ndarray which contains two
        flattened matrices, the first one with u weights and the second one
        with w weights (FCM). The offset for w weights is equal to
        previous_considered_indices.size * space_dim.
        The second argument is obligatory, corresponds to solution index
        and in this case is unused.

    """
    w_matrix_offset = previous_considered_indices.size * space_dim
    max_previous_index = previous_considered_indices.max()

    def fitness_func(solution, _):
        u_matrix = solution[:w_matrix_offset].reshape(-1, space_dim)
        w_matrix = solution[w_matrix_offset:].reshape(-1, space_dim)
        # Note that targets for which we calculate predictions are
        # series[max_previous_index:series.shape[0]].
        # For every predicion we create an array containing indices
        # of its input elements.
        input_indices = (
            np.arange(max_previous_index, series.shape[0])[:, np.newaxis].repeat(
                previous_considered_indices.size, axis=1
            )
            - previous_considered_indices
        )
        # Now we compute matrix containing a-values for each target.
        # The number of columns is equal to space_dim (each target element
        # corresponds to single row which contains a-value for each coordinate).
        a_matrix = _g_func((series[input_indices] * u_matrix).sum(axis=1))
        # We calculate predicted values for each target...
        y_matrix = _sigmoid(np.matmul(a_matrix, w_matrix.T))
        # and the original ones.
        target_matrix = series[max_previous_index:]
        # The last step - error calculation.
        sse = ((y_matrix - target_matrix) ** 2).sum()
        # Note that PyGAD maximizes fitness function.
        return -sse

    return fitness_func


def create_fcm(series, previous_considered_indices):
    """
    Creates fcm for given multidimensional series.

    Parameters
    ----------
    series : numpy.ndarray
        Series for which we want to create FCM. Important note: the series
        must be normalized to [0, 1] interval (all coordinates must
        belong to it). The argument must be two dimensional.
    previous_considered_indices : numpy.ndarray
        Array containing indices of previous elements which will be
        FCM's input for predicting the next one. For example, if you want
        to predict next element using the current one, the previous one and
        before the previous one you can pass numpy.array([1, 2, 3]).
        This argument's entries do not have to be sorted.

    Returns
    -------
    numpy.ndarray
        FCM as two dimensional numpy array.

    """
    space_dim = series.shape[1]
    trained_array_size = previous_considered_indices.size * space_dim + space_dim ** 2
    fitness_func = _create_fitness_func(series, previous_considered_indices, space_dim)

    ga_instance = pygad.GA(
        # pylint: disable=W0511
        # TODO: estimate num_generations in terms of input parameters
        num_generations=3000,
        sol_per_pop=100,
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

    w_matrix_offset = previous_considered_indices.size * space_dim

    return solution[w_matrix_offset:].reshape(space_dim, -1)


if __name__ == "__main__":
    example_series = np.array([0, 0.5, 1] * 100)[:, np.newaxis].repeat(3, axis=1)
    example_series[:, 1] = example_series[:, 1] / 3
    example_series[:, 2] = ((example_series[:, 2] + 0.5) % 1) ** 2
    example_previous_indices = np.r_[1:7]
    fcm = create_fcm(example_series, example_previous_indices)
