"""
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-5*x))

def g_func(x):
    return sigmoid(x)

#previous_considered_indices na ten moment musi być np.ndarray bo series[i - ...]
def _create_fitness_func(series, previous_considered_indices, space_dim):
    w_matrix_offset = previous_considered_indices.size * space_dim
    max_previous_index = previous_considered_indices.max()
    def fitness_func(trained_array):
        u_matrix = trained_array[:w_matrix_offset].reshape(-1, space_dim)
        w_matrix = trained_array[w_matrix_offset:].reshape(-1, space_dim)
        #Nasze targety mają indeksy z range(max_previous_index, series.shape[0])
        #Tworzę tablicę zawierającą dla każdego tego elementu listę indeksów
        #jego punktów szeregów wejciowych
        input_indices = (
            np.arange(max_previous_index, series.shape[0])
                [:, np.newaxis].repeat(previous_considered_indices.size, axis = 1)
                - previous_considered_indices
            )      
        a_matrix = (series[input_indices]*u_matrix).sum(axis = 1)

        #Wartosci w a sa poukladane wierszowo, czyli sa 3 kolumny z a na kazdej wspolrzednej
        y_matrix = np.matmul(a_matrix, w_matrix.T)
        #Teraz w każdym wierszu mamy odpowiadające wyjcie z FCMa
        target_matrix = series[np.arange(max_previous_index, series.shape[0])]
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
