"""
BinaryClassifierModel model (2 classification classes).
"""
import configparser
from functools import partial
from logging import Logger
from typing import List, Literal, Tuple, Optional

import numpy as np
from geneticalgorithm2 import AlgorithmParams
from geneticalgorithm2 import geneticalgorithm2 as ga

from classifier_pipeline import computing_backend
from classifier_pipeline.dataset import SeriesDataset
from classifier_pipeline.fuzzy_cmeans import CMeansTransformer


class BinaryClassifier:
    """
    Base classifier class
    """

    def __init__(self, config: configparser.ConfigParser, cmeans_transformer: CMeansTransformer, logger: Logger):
        """
        @param config: config to use
        @param cmeans_transformer:
        @param logger:
        """
        self.logger = logger

        self.moving_window_size = config.getint("BaseClassifier", 'MovingWindowSize')
        self.moving_window_stride = config.getint("BaseClassifier", 'MovingWindowStride')

        self.ga_params = {
            "max_num_iteration": config.getint("GeneticAlgorithm", "MaxNumIteration"),
            "population_size": config.getint("GeneticAlgorithm", "PopulationSize"),
            "max_iteration_without_improv": config.getint("GeneticAlgorithm", "MaxIterationWithoutImprov"),
            "mutation_probability": config.getfloat("GeneticAlgorithm", "MutationProbability"),
        }

        self.ga_run_params = {
            "no_plot": config.getboolean("GeneticAlgorithmRun", "NoPlot"),
            "disable_printing": config.getboolean("GeneticAlgorithmRun", "DisablePrinting"),
            "disable_progress_bar": config.getboolean("GeneticAlgorithmRun", "DisableProgressBar"),
        }
        self.cmeans_transformer = cmeans_transformer
        self.is_fitted = False
        self.weights: Optional[List[np.ndarray]] = None

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
                + self.cmeans_transformer.num_centroids ** 2
                + 2 * self.cmeans_transformer.num_centroids
        )
        bounds = np.array([[-1, 1]] * trained_array_size)
        ga_model = ga(
            function=fitness_func,
            dimension=trained_array_size,
            variable_boundaries=bounds,
            variable_type="real",
            function_timeout=30,
            algorithm_parameters=AlgorithmParams(**self.ga_params)
        )
        ga_model.run(
            set_function=ga.set_function_multiprocess(fitness_func),
            stop_when_reached=0,
            **self.ga_run_params
        )
        solution = ga_model.output_dict["variable"]
        self.logger.info(
            f"Fraction of misclassified series for base classifier fitted on {dataset.labels}: "
            f"{ga_model.output_dict['function'] / sum(len(x) for x in memberships)}"
        )

        self.weights = computing_backend.split_weights_array(solution, self.moving_window_size,
                                                             self.cmeans_transformer.num_centroids)
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
        prediction = computing_backend.predict_series_class_idx(
            membership_matrix, *self.weights, self.moving_window_size, self.moving_window_stride)
        return prediction[0], prediction[1]
