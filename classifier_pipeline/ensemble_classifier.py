from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from logging import Logger
from typing import Literal, Tuple, List

import numpy as np

from classifier_pipeline.binary_classifier import BinaryClassifier
from classifier_pipeline.dataset import SeriesDataset
from classifier_pipeline.fuzzy_cmeans import CMeansComputer, CMeansTransformer


class EnsembleClassifier(ABC):

    def build_classifier(self, method: Literal['1_vs_all', 'asymmetric_1_vs_1', 'symmetric_1_vs_1',
                                               'combined_symmetric_1_vs_1'], config: MutableMapping):
        classifiers = {
            '1_vs_all': OneVsAllClassifier,
            'asymmetric_1_vs_1': AsymmetricOneVsOneClassifier,
            'symmetric_1_vs_1': SymmetricOneVsOneClassifier,
            'combined_symmetric_1_vs_1': CombinedOneVsOneClassifier
        }
        return classifiers[method](config)

    def __init__(self, config: MutableMapping, logger: Logger):
        self.config = config
        self.logger = logger
        self.binary_classifiers: Tuple[BinaryClassifier] = None
        self.fcm_concept_count = config["BaseClassifier"]["FCMConceptCount"]

    @abstractmethod
    def fit(self, dataset: SeriesDataset) -> None:
        """

        @param dataset:
        @return:
        """

    @abstractmethod
    def predict(self, series: np.ndarray) -> int:
        """

        @param series:
        @return:
        """

    @abstractmethod
    def evaluate(self, dataset: SeriesDataset) -> float:
        """

        @param dataset:
        @return:
        """


class OneVsAllClassifier(EnsembleClassifier):

    def __init__(self, config: MutableMapping, logger: Logger):
        super().__init__(config)

    def fit(self, dataset: SeriesDataset) -> EnsembleClassifier:
        binary_classifiers: List[BinaryClassifier] = []

        for class_idx in range(dataset.n_classes):
            idx_vs_all_dataset = dataset.make_one_vs_all(class_idx)
            assert idx_vs_all_dataset.get_label(0) != SeriesDataset.ALL_OTHER_LABEL
            assert idx_vs_all_dataset.get_label(1) == SeriesDataset.ALL_OTHER_LABEL
            cmeans_computer = CMeansComputer(self.config)
            centroids = cmeans_computer.compute(np.vstack(idx_vs_all_dataset.get_series_list([0])),
                                                self.fcm_concept_count)
            cmeans_transformer = CMeansTransformer(self.config, centroids)
            assert cmeans_transformer.num_centroids == self.fcm_concept_count

            binary_classifier = BinaryClassifier(self.config, cmeans_transformer, self.logger)
            binary_classifier.fit(idx_vs_all_dataset)
            assert binary_classifier.is_fitted

            binary_classifiers.append(binary_classifier)

        assert len(binary_classifiers) == dataset.n_classes
        self.binary_classifiers = tuple(binary_classifiers)

    def evaluate(self, dataset: SeriesDataset) -> float:
        pass


class AsymmetricOneVsOneClassifier(EnsembleClassifier):
    def fit(self, dataset: SeriesDataset) -> None:

        binary_classifiers: List[BinaryClassifier] = []

        for class_idx_1 in range(dataset.n_classes):
            for class_idx_2 in range(dataset.n_classes):
                if class_idx_1 != class_idx_2:
                    truncated_dataset = dataset.truncate((class_idx_1, class_idx_2))
                    cmeans_computer = CMeansComputer(self.config)
                    centroids = cmeans_computer.compute(np.vstack(truncated_dataset.get_series_list(0)))
                    cmeans_transformer = CMeansTransformer(self.config, centroids)
                    assert cmeans_transformer.num_centroids == self.fcm_concept_count

                    binary_classifier = BinaryClassifier(self.config, cmeans_transformer, self.logger)
                    binary_classifier.fit(truncated_dataset)
                    assert binary_classifier.is_fitted

                    binary_classifiers.append(binary_classifier)

        assert 2 * len(binary_classifiers) == dataset.n_classes * (dataset.n_classes - 1)
        self.binary_classifiers = binary_classifiers


class SymmetricOneVsOneClassifier(EnsembleClassifier):
    def __init__(self, config: MutableMapping, logger: Logger):
        super().__init__(config, logger)
        if (self.fcm_concept_count % 2) != 0:
            raise ValueError(
                f"Concept count for symmetric one vs one classifier must be even, got {self.fcm_concept_count}")

    def fit(self, dataset: SeriesDataset) -> None:
        binary_classifiers: List[BinaryClassifier] = []

        for class_idx_1 in range(dataset.n_classes):
            for class_idx_2 in range(class_idx_1 + 1, dataset.n_classes):
                truncated_dataset = dataset.truncate((class_idx_1, class_idx_2))
                cmeans_computer = CMeansComputer(self.config)
                centroids_1 = cmeans_computer.compute(np.vstack(truncated_dataset.get_series_list(0)), self.fcm_concept_count // 2)
                centroids_2 = cmeans_computer.compute(np.vstack(truncated_dataset.get_series_list(1)), self.fcm_concept_count // 2)
                centroids = np.vstack([centroids_1, centroids_2])
                cmeans_transformer = CMeansTransformer(self.config, centroids)
                assert cmeans_transformer.num_centroids == self.fcm_concept_count

                binary_classifier = BinaryClassifier(self.config, cmeans_transformer, self.logger)
                binary_classifier.fit(truncated_dataset)
                assert binary_classifier.is_fitted

                binary_classifiers.append(binary_classifier)

        assert len(binary_classifiers) == dataset.n_classes * (dataset.n_classes - 1)
        self.binary_classifiers = binary_classifiers


class CombinedOneVsOneClassifier(EnsembleClassifier):
    def fit(self, dataset: SeriesDataset) -> None:
        binary_classifiers: List[BinaryClassifier] = []

        for class_idx_1 in range(dataset.n_classes):
            for class_idx_2 in range(class_idx_1 + 1, dataset.n_classes):
                truncated_dataset = dataset.truncate((class_idx_1, class_idx_2))
                cmeans_computer = CMeansComputer(self.config)
                both_classes_series_list = truncated_dataset.get_series_list(0) + truncated_dataset.get_series_list(1)
                centroids = cmeans_computer.compute(np.vstack(both_classes_series_list), self.fcm_concept_count)
                cmeans_transformer = CMeansTransformer(self.config, centroids)
                assert cmeans_transformer.num_centroids == self.fcm_concept_count

                binary_classifier = BinaryClassifier(self.config, cmeans_transformer, self.logger)
                binary_classifier.fit(truncated_dataset)
                assert binary_classifier.is_fitted

                binary_classifiers.append(binary_classifier)

        assert len(binary_classifiers) == dataset.n_classes * (dataset.n_classes - 1)
        self.binary_classifiers = binary_classifiers
