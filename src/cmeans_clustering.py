import functools
from itertools import accumulate
import numpy as np
import skfuzzy as fuzz


def _include_diffs(series_list):
    """
    series_list (nr_of_series, 3, series_len)

    returns:
    combined_coordinates_list (nr_of_series, 6, series_len)
    lengths (nr_of_series)
    """
    combined_coordinates_list = []
    for series in series_list:
        adjacent_differences = np.diff(series)
        series = series[:, 1:]
        combined_coordinates = np.hstack((series, adjacent_differences)).reshape(
            -1, series.shape[1]
        )
        combined_coordinates_list.append(combined_coordinates)
    return combined_coordinates_list


def _accumulate_series(series_list):
    """
    series_list (nr_of_series, 3, series_len)

    returns:
    horizontal_combined_coordinates (6, sum_of_series_len)
    lengths (nr_of_series)
    """
    combined_coordinates_list = _include_diffs(series_list)
    lengths = [x.shape[1] for x in combined_coordinates_list]
    horizontal_combined_coordinates = np.hstack(combined_coordinates_list)

    return horizontal_combined_coordinates, lengths


def _cmeans_wrapper(coordinates, concept_count):
    """
    coordinates ( 6, sum_of_series_len)

    returns:
    tuple((concept_count, 6), (sum_of_series_len,concept_count))
    """
    m = 2
    error = 1e-8
    maxiter = 1e2
    ret = fuzz.cmeans(coordinates, concept_count, m, error, maxiter)
    return (
        ret[0],
        ret[1].T,
    )


def find_clusters(series_list, concept_count):
    """
    series_list (nr_of_series, 3, series_len)

    returns:
    clusters (concept_count, 6)
    series_memberships (nr_of_series, series_len, concept_count)
    """
    horizontal_combined_coordinates, lengths = _accumulate_series(series_list)
    clusters, memberships_combined = _cmeans_wrapper(
        horizontal_combined_coordinates, concept_count=concept_count
    )
    spliting_indices = list(accumulate(lengths))[:-1]
    series_memberships = np.split(memberships_combined, spliting_indices)
    return clusters, series_memberships


def find_memberships(series_list, centroids):
    """
    series_list (nr_of_series, 3, series_len)
    centroids - 2d numpy.ndarray with centroids

    returns:
    list of 2d numpy.ndarray
    (nr_of_series, series_len, concept_count)
    """
    predict_fun = functools.partial(
        fuzz.cmeans_predict, cntr_trained=centroids, m=2, error=1e-8, maxiter=1e2
    )
    with_diffs = _include_diffs(series_list)
    return [x[1].T for x in map(predict_fun, with_diffs)]
