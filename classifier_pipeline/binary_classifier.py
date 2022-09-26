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
from classifier_pipeline.fuzzy_cmeans import CMeansTransformer
from classifier_pipeline.dataset import SeriesDataset


class BinaryClassifier:
    """
    Base classifier class
    """
    def __init__(self, config: MutableMapping, cmeans_transformer: CMeansTransformer, logger: Logger):
        """
        @param config: config to use
        @param cmeans_transformer:
        @param logger:
        """
        self.logger = logger

        base_classifier_config = config["BaseClassifier"]
        self.moving_window_size = base_classifier_config['MovingWindowSize']
        self.moving_window_stride = base_classifier_config['MovingWindowStride']

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
        self.cmeans_transformer = cmeans_transformer
        self.is_fitted = False
        self.weights: List[np.ndarray] = None

    def fit(self, dataset: SeriesDataset) -> "BinaryClassifier":
        """
        Fits classifier
        @param dataset:
        @return: self
        """
        if dataset.n_classes != 2:
            raise ValueError("Binary classifier accepts only two-class datasets")

        memberships = dataset.transform(self.cmeans_transformer.transform)
        fitness_func = partial(computing_backend.fitness_func, membership_matrices=memberships,
                               moving_window_size=self.moving_window_size,
                               moving_window_stride=self.moving_window_stride)

        trained_array_size = (
                self.moving_window_size * self.cmeans_transformer.num_centroids
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

    def predict(self, series_array: np.ndarray) -> Tuple[Literal[0, 1], np.ndarray]:
        """
        Predicts class index for given series
        @param series_array:
        @return: tuple with class index (0 or 1) and ndarray with two scores for each class
        """
        if not self.is_fitted:
            raise RuntimeError("Base classifier has not been fitted")

        membership_matrix = self.cmeans_transformer.transform(series_array)
        prediction = computing_backend.predict_series_class_idx(membership_matrix, *self._uwv_matrices,
                                                                self._previous_considered_indices, self._move)
        return prediction[0], prediction[1]
