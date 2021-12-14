"""
C-means tools.
"""

import functools
from itertools import accumulate
import numpy as np
import skfuzzy as fuzz


def _include_diffs(series_list):
    """
    Parameters
    ----------
    series_list : list
        (nr_of_series, nr_of_coordinates, series_len)

    Returns
    -------
    list
        (nr_of_series, 2 * nr_of_coordinates, series_len)

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
    Parameters
    ----------
    series_list : list
        (nr_of_series, nr_of_coordinates, series_len)

    Returns
    -------
    numpy.ndarray
        (2 * nr_of_coordinates, sum_of_series_len)
    list
        (nr_of_series)

    """
    combined_coordinates_list = _include_diffs(series_list)
    lengths = [x.shape[1] for x in combined_coordinates_list]
    horizontal_combined_coordinates = np.hstack(combined_coordinates_list)

    return horizontal_combined_coordinates, lengths


def _cmeans_wrapper(coordinates, concept_count):
    """
    Parameters
    ----------
    coordinates : numpy.ndarray
        (2 * nr_of_coordinates, sum_of_series_len)
    concept_count : int

    Returns
    -------
    tuple
        ((concept_count, 2 * nr_of_coordinates), (sum_of_series_len, concept_count))

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
    Parameters
    ----------
    series_list : list
        (nr_of_series, nr_of_coordinates, series_len)
    concept_count : int

    Returns
    -------
    list
        (concept_count, 2 * nr_of_coordinates)
    list
        (nr_of_series, series_len, concept_count)

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

    Parameters
    ----------
    series : numpy.ndarray
        (2 * nr_of_coordinates, series_len)
    centroids : numpy.ndarray
        (concept_count, 2 * nr_of_coordinates)

    Returns
    -------
    numpy.ndarray
        (series_len, concept_count)

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
    Parameters
    ----------
    series_list : list
        (nr_of_series, nr_of_coordinates, series_len)
    centroids : numpy.ndarray
        centroids

    Returns
    -------
    list
        (nr_of_series, series_len, concept_count)

    """

    # predict_fun = functools.partial(_find_memberships_for_series, centroids=centroids)
    with_diffs = _include_diffs(series_list)

    # memberships = list(map(predict_fun, with_diffs))
    predict_fun = functools.partial(
        fuzz.cmeans_predict, cntr_trained=centroids, m=2, error=1e-8, maxiter=2
    )

    return [x[0].T for x in map(predict_fun, with_diffs)]


if __name__ == "__main__":
    mainS = np.array(
        [
            [1, 1, 2, 1],
            [2, 2, 3, 2],
            [3, 3, 4, 3],
            [4, 4, 5, 4],
            [5, 5, 6, 5],
            [6, 6, 7, 6],
        ]
    )
    res = fuzz.cmeans(mainS, 2, 2, 1e-8, 1e2)
    test_centroids, test_membership = res[0], res[1]
    print(test_centroids)
    print(test_membership.T)
    # centroids = np.array([[0, 1, 2, 3, 4, 5], np.r_[2:8]])
    print(_find_memberships_for_series(mainS, test_centroids))
    print(fuzz.cmeans_predict(mainS, test_centroids, 2, 1e-8, 1e2)[0].T)
    print(
        np.array_equal(
            _find_memberships_for_series(mainS, test_centroids), test_membership.T
        )
    )
    print(
        np.array_equal(
            fuzz.cmeans_predict(mainS, test_centroids, 2, 1, 2)[0].T, test_membership.T
        )
    )
