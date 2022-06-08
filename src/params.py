from geneticalgorithm2 import AlgorithmParams

_GA_PARAMS = {
    "variable_type": "real",
    "algorithm_parameters": AlgorithmParams(
        max_num_iteration=20,
        population_size=10,
        max_iteration_without_improv=50,
        mutation_probability=0.05,
    ),
}
_GA_RUN_PARAMS = {
    "no_plot": True,
    "disable_printing": True,
    "disable_progress_bar": True,
}

cmeans_params = {
    "m": 2,
    "error": 1e-8,
    "maxiter": 1e2
}

method = 0
# 0 - classifier k vs l with centroids for k
# 1 - classifier k vs l with centroids for k and l trained separately
# 2 - classifier k vs all with centroids for k
# 3 - classifier k vs l with centroids for k and l trained together