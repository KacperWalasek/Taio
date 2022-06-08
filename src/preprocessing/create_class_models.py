import os
import functools
from sys import platform
import preprocessing.read_data as read_data
import preprocessing.cmeans_clustering as cmeans_clustering
import params

available_cpus = (
    int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    if "SLURM_JOB_ID" in os.environ
    else os.cpu_count()
)


from multiprocessing import Pool
    
def create_class_models(dir_path, previous_considered_indices, concept_count):
    class_dirs = [(entry.name, entry.path) for entry in os.scandir(dir_path) if entry.is_dir()]

    fun = functools.partial(_read_and_cluster_class_series, concept_count=concept_count)
    with Pool(min(available_cpus, len(class_dirs))) as p:
        class_models = p.map(fun, class_dirs)

    too_short_series = next(
        (
            (series, class_model[0])
            for class_model in class_models
            for series in class_model[1]
            if series.shape[1] < max(previous_considered_indices) + 2
        ),
        None,
    )
    if too_short_series is not None:
        raise RuntimeError(
            "Specified previous_considered_indices array invalid for given dataset"
            f" - one of the series in class {too_short_series[1]}"
            " after preprocessing with read_data module has only"
            f" {too_short_series[0].shape[1]} elements."
            " Remember that c-means processed series has one element less"
            " due to adding adjacent differences to points (the first one is discarded)."
        )
    return class_models


def _read_and_cluster_class_series(class_dir, concept_count):
    series_list = []
    for file in os.scandir(class_dir[1]):
        if file.name.endswith(".csv"):
            series_list.append(read_data.process_data(file.path, 1))

    if params.method==3:
        clusters = []
    else:
        clusters = cmeans_clustering.find_clusters(series_list, concept_count)
    
    return (
        class_dir[0],
        series_list,
        clusters,
    )
