"""
BinaryClassifierModel model (2 classification classes).
"""
from collections.abc import MutableMapping
from functools import partial
from logging import Logger
from typing import List, Literal, Tuple

import numpy as np
import skfuzzy as fuzz
from geneticalgorithm2 import AlgorithmParams
from geneticalgorithm2 import geneticalgorithm2 as ga

from classifier_pipeline import computing_backend
from classifier_pipeline.dataset import SeriesDataset


class UncombinedBinaryClassifierBase:
    """
    Class representing single class vs class classifier_pipeline model.

    Parameters
    ----------
    class_numbers : tuple
        Tuple containing two class numbers.
    membership_matrices : tuple
        Tuple containing two lists of series membership matrices.
        Both elements of tuple correspond to classes specified in class_numbers.
    centroids: numpy.ndarray
        Ndarray of shape (concept_count, centroids_space_dimension) containing
        coordinates of centroids.
    previous_considered_indices: list
        List containing indices of previous elements which will be
        FCM's input for predicting the next one.
    move: int
        Step of frame used in processing single series.

    """

    def __init__(self, fuzzy_space_class_idx: Literal[0, 1], config: MutableMapping, logger: Logger):
        self.logger = logger

        self.fuzzy_space_class_idx = fuzzy_space_class_idx

        base_classifier_config = config["BaseClassifier"]
        self.fcm_concept_count = base_classifier_config['FCMConceptCount']
        self.moving_window_size = base_classifier_config['MovingWindowSize']
        self.moving_window_stride = base_classifier_config['MovingWindowStride']

        cmeans_config = config["FuzzyCMeans"]
        self.cmeans_params = {
            "m": cmeans_config["M"], "error": cmeans_config["Error"], "maxiter": cmeans_config["Maxiter"]
        }
        ga_config = config["GeneticAlgorithm"]
        self.ga_params = {
            "max_num_iteration": ga_config["MaxNumIteration"],
            "population_size": ga_config["PopulationSize"],
            "max_iteration_without_improv": ga_config["MaxIterationWithoutImprov"],
            "mutation_probability": ga_config["MutationProbability"]
        }
        ga_run_config = config["GeneticAlgorithmRun"]
        self.ga_run_params = {
            "no_plot": ga_run_config["NoPlot"],
            "disable_printing": ga_run_config["DisablePrinting"],
            "disable_progress_bar": ga_run_config["DisableProgressBar"],
        }
        self.is_fitted = False

        self.weights: List[np.ndarray] = None
        self.centroids = None

    def fit(self, dataset: SeriesDataset, combine_concepts: bool):
        if dataset.n_classes != 2:
            raise ValueError("Binary classifier accepts only two-class datasets")

        reference_class_series_list = dataset.get_series_list(self.fuzzy_space_class_idx)
        self.centroids = self.compute_centroids(np.vstack(reference_class_series_list))

        memberships = dataset.transform(partial(self.predict_memberships, centroids=self.centroids))
        fitness_func = partial(computing_backend.fitness_func, membership_matrices=memberships,
                               moving_window_size=self.moving_window_size,
                               moving_window_stride=self.moving_window_stride)

        trained_array_size = (
                self.moving_window_size * self.fcm_concept_count
                + self.fcm_concept_count ** 2
                + 2 * self.fcm_concept_count
        )
        bounds = np.array([[-1, 1]] * trained_array_size)
        ga_model = ga(
            function=fitness_func,
            dimension=trained_array_size,
            variable_boundaries=bounds,
            algorithm_parameters=AlgorithmParams(**self.ga_params)
        )
        ga_model.run(
            set_function=ga.set_function_multiprocess(fitness_func),
            stop_when_reached=0,
            variable_type="real",
            function_timeout=30,
            **self.ga_run_params
        )
        solution = ga_model.output_dict["variable"]
        self.logger.info(
            f"Fraction of misclassified series for base classifier fitted on {dataset.labels}: "
            f"{ga_model.output_dict['function'] / sum(len(x) for x in memberships)}"
        )

        self.weights = computing_backend.split_weights_array(solution, self.moving_window_size, self.fcm_concept_count)
        self.is_fitted = True
        return self

    def fit_centroids(self, points: np.ndarray, num_centroids: int) -> np.ndarray:
        """
        Computes centroids for a given array of datapoints
        @param points: array of shape (num_points, dimension)
        @param num_centroids: the number of centroids to use
        @return: array of shape (num_centroids, dimension)
        """
        centroids = fuzz.cmeans(points.T, num_centroids, **self.cmeans_params)[0]
        return centroids

    def predict_memberships(self, points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Computes membership degrees of given points to given centroids
        @param points: array of shape (num_points, dimension)
        @param centroids: array of shape (num_centroids, dimension)
        @return: array of shape (num_points, num_centroids)
        """
        return fuzz.cmeans_predict(points.T, centroids, **self.cmeans_params)[0].T

    def predict(self, series_array: np.ndarray) -> Tuple[int, float]:
        if not self.is_fitted:
            raise RuntimeError("Base classifier has not been fitted")

        membership_matrix = self.fit_centroids()
        prediction = computing_backend.predict_series_class_idx(membership_matrix, *self._uwv_matrices,
                                                                self._previous_considered_indices, self._move)
        return prediction[0], prediction[1]
