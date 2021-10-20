"""
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-5*x))

def g_func(x):
    return sigmoid(x)

#previous_considered_indices na ten moment musi być np.ndarray bo series[i - ...]
def create_fitness_func(series, previous_considered_indices, space_dim):
    w_matrix_offset = previous_considered_indices.size * space_dim
    max_previous_index = previous_considered_indices.max()
    def fitness_func(trained_array):
        u_matrix = trained_array[:w_matrix_offset].reshape(-1, space_dim)
        w_matrix = trained_array[w_matrix_offset:].reshape(-1, space_dim)
        inputs = np.array(map(lambda val: val - ))
        #Nasze targety mają indeksy z range(max_previous_index, series.shape[0])
        #Tworzę tablicę zawierającą dla każdego tego elementu listę indeksów
        #jego punktów szeregów wejciowych
        input_indices = (
            np.arange(max_previous_index, series.shape([0]))
                [:, np.newaxis].repeat(previous_considered_indices.size, axis = 1)
                - previous_considered_indices
            )
        def compute_a_values(inputs):
            return (u_matrix * inputs).sum(axis = 0)
        a_matrix = np.apply_along_axis(compute_a_values, 1, series[input_indices])
        #Wartosci w a sa poukladane wierszowo, czyli sa 3 kolumny z a na kazdej wspolrzednej
        #TODO niżej transpozycje wyciagnac na zewnątrz
        y_matrix = np.matmul(w_matrix, a_matrix.transpose()).transpose()
        #Teraz w każdym wierszu mamy odpowiadające wyjcie z FCMa
        target_matrix = series[np.arange(max_previous_index, series.shape([0]))]
        sse = (y_matrix - target_matrix)**2
        return sse
    return fitness_func
            

def create_fcm(series, previous_considered_indices):
    space_dim = series.shape[1]
    u_matrix_bounds = np.array([[-1, 1]] * (previous_considered_indices.size * space_dim))
    w_matrix_bounds = np.array([[-1, 1]] * space_dim**2)
    all_bounds = np.vstack(u_matrix_bounds, w_matrix_bounds)
