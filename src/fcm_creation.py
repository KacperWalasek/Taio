"""
"""
import numpy as np

from geneticalgorithm2 import geneticalgorithm2 as ga # for creating and running optimization model

from geneticalgorithm2 import Generation, AlgorithmParams # classes for comfortable parameters setting and getting

from geneticalgorithm2 import Crossover, Mutations, Selection # classes for specific mutation and crossover behavior

from geneticalgorithm2 import Population_initializer # for creating better start population

from geneticalgorithm2 import np_lru_cache # for cache function (if u want)

from geneticalgorithm2 import plot_pop_scores # for plotting population scores, if u want

from geneticalgorithm2 import Callbacks # simple callbacks

from geneticalgorithm2 import Actions, ActionConditions, MiddleCallbacks # middle callbacks

def create_fcm(series, previous_considered_indices):
    space_dim = series.shape[1]
    u_matrix_bounds = np.array([[-1, 1]] * (previous_considered_indices.size * space_dim))
    w_matrix_bounds = np.array([[-1, 1]] * space_dim**2)
    all_bounds = np.vstack(u_matrix_bounds, w_matrix_bounds)

test = np.array([[0, 10]] * 3**2)
def function(x):
    return np.sum(x)
model = ga(function, dimension = 9, 
                variable_type='real', 
                 variable_boundaries = test,
                 variable_type_mixed = None, 
                 function_timeout = 10,
                 algorithm_parameters=AlgorithmParams()
            )   
model.run(
    no_plot = False, 
    disable_progress_bar = False,
    disable_printing = False,

    set_function = None, 
    apply_function_to_parents = False, 
    start_generation = {'variables':None, 'scores': None},
    studEA = False,
    mutation_indexes = None,

    init_creator = None,
    init_oppositors = None,
    duplicates_oppositor = None,
    remove_duplicates_generation_step = None,
    revolution_oppositor = None,
    revolution_after_stagnation_step = None,
    revolution_part = 0.3,
    
    population_initializer = Population_initializer(select_best_of = 1, local_optimization_step = 'never', local_optimizer = None),
    
    stop_when_reached = None,
    callbacks = [],
    middle_callbacks = [],
    time_limit_secs = None, 
    save_last_generation_as = None,
    seed = None
    )
