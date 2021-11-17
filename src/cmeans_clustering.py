"""
C-means tools.
"""

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
    maxiter = 1e6
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


def _find_memberships_for_series(series, centroids):
    """
    Get memberships for single time series.

    series (6, series_len)
    centroids (concept_count, 6)

    returns:
    ndarray (series_len, concept_count)
    """

    series = series.T
    c = np.shape(centroids)[0]
    series = np.repeat(series[:, np.newaxis, :], c, axis=1)
    centroids = np.repeat(centroids[np.newaxis, :, :], np.shape(series)[0], axis=0)
    distances = np.fmax(
        np.linalg.norm(series - centroids, axis=2).T, np.finfo(np.float64).eps
    )
    distances_squared = distances ** 2
    return (
        1
        / np.sum(
            np.repeat(distances_squared[np.newaxis, :, :], c, axis=0)
            / distances_squared[:, np.newaxis, :],
            axis=0,
        )
    ).T


def find_memberships(series_list, centroids):
    """
    series_list (nr_of_series, 3, series_len)
    centroids - 2d numpy.ndarray with centroids

    returns:
    list of 2d numpy.ndarray
    (nr_of_series, series_len, concept_count)
    """

    predict_fun = functools.partial(_find_memberships_for_series, centroids=centroids)
    with_diffs = _include_diffs(series_list)

    memberships = list(map(predict_fun, with_diffs))
    return memberships


if __name__ == "__main__":
    s = np.array(
        [
            [1, 1, 2, 1],
            [2, 2, 3, 2],
            [3, 3, 4, 3],
            [4, 4, 5, 4],
            [5, 5, 6, 5],
            [6, 6, 7, 6],
        ]
    )
    c = np.array([[0, 1, 2, 3, 4, 5], np.r_[2:8]])
    print(_find_memberships_for_series(s, c))
    print(fuzz.cmeans_predict(s, c, 2, 1e-8, 1e2)[0])
