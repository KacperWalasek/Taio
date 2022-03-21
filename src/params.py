
_GA_RUN_PARAMS = {
    "no_plot": True,
    "disable_printing": False,
    "disable_progress_bar": False,
}

cmeans_params = {
    "m": 2,
    "error": 1e-8,
    "maxiter": 1e4
}

method = 0
# 0 - classifier k vs l with centroids for k
# 1 - classifier k vs l with centroids for k and l
# 2 - classifier k vs all with centroids for k
