"""
This module contains create_fcm function which should be used to create
fuzzy cognitive map for given series. Under the hood, genetic
algorithm is used.
"""
import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import AlgorithmParams

def sigmoid(x):
    """
    Sigmoid function with parameter 5 compatibile with numpy.ndarray.
    """
    return 1 / (1 + np.exp(-5*x))

def _g_func(x):
    """
    Internal function used to modify the outputs of first training layer
    (the one with u weights).
    """
    return sigmoid(x)

#previous_considered_indices na ten moment musi być np.ndarray bo series[i - ...]
def _create_fitness_func(series, previous_considered_indices, space_dim):
    """
    Creates fitness function for genetic algorithm.

    Parameters
    ----------
    series : numpy.ndarray
        Series for which we want to create FCM. Important note: the series
        must be normalized to [0, 1] interval (all coordinates must
        belong to it).
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
        Fitness function ready to be used in genetic algorithm. Its only
        parameter is a single dimensional numpy.ndarray which contains two
        flattened matrices, the first one with u weights and the second one
        with w weights (FCM). The offset for w weights is equal to
        previous_considered_indices.size * space_dim.

    """
    w_matrix_offset = previous_considered_indices.size * space_dim
    max_previous_index = previous_considered_indices.max()
    def fitness_func(trained_array):
        #Gathering and reshaping both weights' matrices from
        #single-dimensional input
        u_matrix = trained_array[:w_matrix_offset].reshape(-1, space_dim)
        w_matrix = trained_array[w_matrix_offset:].reshape(-1, space_dim)
        #Note that targets for which we calculate predictions are
        #series[max_previous_index:series.shape[0]].
        #For each one we create array containing indices of its' input elements.
        input_indices = np.arange(max_previous_index, series.shape[0])[:, np.newaxis].repeat(previous_considered_indices.size, axis = 1) - previous_considered_indices    
        #Now we compute matrix containing a values for each target.
        #The number of columns is equal to space_dim (each target element
        #corresponds to single row which contains a values for each coordinate).
        a_matrix = _g_func((series[input_indices]*u_matrix).sum(axis = 1))
        #We calculate predicted values for each target...
        y_matrix = sigmoid(np.matmul(a_matrix, w_matrix.T))
        #and the original ones.
        target_matrix = series[max_previous_index:]
        #The last step - error calculation.
        sse = ((y_matrix - target_matrix)**2).sum()
        return sse
    return fitness_func
            

def create_fcm(series, previous_considered_indices):
    space_dim = series.shape[1]
    trained_array_size = previous_considered_indices.size * space_dim + space_dim**2
    trained_array_bounds = np.array([[-1, 1]] * trained_array_size)
    fitness_func = _create_fitness_func(series, previous_considered_indices, space_dim)
    model = ga(fitness_func, dimension = trained_array_size, 
                variable_type='real', 
                 variable_boundaries = trained_array_bounds,
                 variable_type_mixed = None, 
                 function_timeout = 10,
                 algorithm_parameters=AlgorithmParams(
                     max_num_iteration = None,
                     population_size = 100,
                     mutation_probability = 0.1,
                     elit_ratio = 0.01,
                     crossover_probability = 0.5,
                     parents_portion = 0.3,
                     crossover_type = 'uniform',
                     mutation_type = 'uniform_by_center',
                     selection_type = 'roulette',
                     max_iteration_without_improv = None
                     )
            )
    model = ga(fitness_func, dimension = trained_array_size, 
                variable_type='real', 
                 variable_boundaries = trained_array_bounds,
                 variable_type_mixed = None, 
                 function_timeout = 10,
                 algorithm_parameters=AlgorithmParams()
            )
    model.run()
