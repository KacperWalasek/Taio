import configparser
from abc import ABC, abstractmethod
from logging import Logger
from typing import Literal, Tuple, List, NamedTuple, Optional

import numpy as np

from classifier_pipeline.binary_classifier import BinaryClassifier
from classifier_pipeline.dataset import SeriesDataset
from classifier_pipeline.fuzzy_cmeans import CMeansComputer, CMeansTransformer

BinaryClassifierItem = NamedTuple('BinaryClassifierItem', [('model', BinaryClassifier),
                                                           ('reference_classes', Tuple[Optional[int], Optional[int]])])


class EnsembleClassifier(ABC):

    @staticmethod
    def build_classifier(method: Literal['1_vs_all', 'asymmetric_1_vs_1', 'symmetric_1_vs_1',
                                         'combined_symmetric_1_vs_1'], config: configparser.ConfigParser,
                         logger: Logger):
        classifiers = {
            '1_vs_all': OneVsAllClassifier,
            'asymmetric_1_vs_1': AsymmetricOneVsOneClassifier,
            'symmetric_1_vs_1': SymmetricOneVsOneClassifier,
            'combined_symmetric_1_vs_1': CombinedOneVsOneClassifier
        }
        return classifiers[method](config, logger)

    def __init__(self, config: configparser.ConfigParser, logger: Logger):
        self.config = config
        self.logger = logger
        self.binary_classifiers: Optional[Tuple[BinaryClassifierItem, ...]] = None
        self.fcm_concept_count = config.getint("BaseClassifier", "FCMConceptCount")
        self.n_classes: Optional[int] = None

    @abstractmethod
    def fit(self, dataset: SeriesDataset) -> None:
        """

        @param dataset:
        @return:
        """

    def predict(self, series: np.ndarray) -> int:
        """

        @param series:
        @return:
        """
        series_class_votes = np.zeros(self.n_classes, dtype=np.int32)
        series_class_weights = np.zeros(self.n_classes)
        for binary_classifier, reference_classes in self.binary_classifiers:
            predicted_idx, predicted_weights = binary_classifier.predict(series)
            if (predicted_class := reference_classes[predicted_idx]) is not None:
                series_class_votes[predicted_class] += 1
            for i, predicted_weight in enumerate(predicted_weights):
                if reference_classes[i] is not None:
                    series_class_weights[reference_classes[i]] += predicted_weight
        # Below lines work for both situations:
        # 1. There is only one max_votes_index
        # 2. There is more than one max_votes_index, then choose among them the one with maximum weight
        max_votes_indices = np.flatnonzero(series_class_votes == series_class_votes.max())
        return max_votes_indices[np.argmax(series_class_weights[max_votes_indices])].item()

    def evaluate(self, dataset: SeriesDataset) -> float:
        """

        @param dataset:
        @return:
        """
        num_series = 0
        num_correctly_classified = 0
        for i in range(dataset.n_classes):
            series_list = dataset.get_series_list(i)
            for series in series_list:
                num_series += 1
                predicted_class = self.predict(series)
                if predicted_class == i:
                    num_correctly_classified += 1
        return num_correctly_classified / num_series


class OneVsAllClassifier(EnsembleClassifier):

    def fit(self, dataset: SeriesDataset) -> None:
        binary_classifiers: List[BinaryClassifierItem] = []

        for class_idx in range(dataset.n_classes):
            idx_vs_all_dataset = dataset.make_one_vs_all(class_idx)
            assert idx_vs_all_dataset.get_label(0) != SeriesDataset.ALL_OTHER_LABEL
            assert idx_vs_all_dataset.get_label(1) == SeriesDataset.ALL_OTHER_LABEL
            cmeans_computer = CMeansComputer(self.config)
            centroids = cmeans_computer.compute(np.vstack(idx_vs_all_dataset.get_series_list(0)),
                                                self.fcm_concept_count)
            cmeans_transformer = CMeansTransformer(self.config, centroids)
            assert cmeans_transformer.num_centroids == self.fcm_concept_count

            binary_classifier = BinaryClassifier(self.config, cmeans_transformer, self.logger)
            binary_classifier.fit(idx_vs_all_dataset)
            assert binary_classifier.is_fitted

            binary_classifiers.append(
                BinaryClassifierItem(model=binary_classifier, reference_classes=(class_idx, None)))

        assert len(binary_classifiers) == dataset.n_classes
        self.n_classes = dataset.n_classes
        self.binary_classifiers = tuple(binary_classifiers)


class AsymmetricOneVsOneClassifier(EnsembleClassifier):
    def fit(self, dataset: SeriesDataset) -> None:

        binary_classifiers: List[BinaryClassifierItem] = []

        for class_idx_1 in range(dataset.n_classes):
            for class_idx_2 in range(dataset.n_classes):
                if class_idx_1 != class_idx_2:
                    truncated_dataset = dataset.truncate((class_idx_1, class_idx_2))
                    cmeans_computer = CMeansComputer(self.config)
                    centroids = cmeans_computer.compute(np.vstack(truncated_dataset.get_series_list(0)),
                                                        self.fcm_concept_count)
                    cmeans_transformer = CMeansTransformer(self.config, centroids)
                    assert cmeans_transformer.num_centroids == self.fcm_concept_count

                    binary_classifier = BinaryClassifier(self.config, cmeans_transformer, self.logger)
                    binary_classifier.fit(truncated_dataset)
                    assert binary_classifier.is_fitted

                    binary_classifiers.append(
                        BinaryClassifierItem(model=binary_classifier, reference_classes=(class_idx_1, class_idx_2)))

        assert len(binary_classifiers) == dataset.n_classes * (dataset.n_classes - 1)
        self.binary_classifiers = tuple(binary_classifiers)


class SymmetricOneVsOneClassifier(EnsembleClassifier):
    def __init__(self, config: configparser.ConfigParser, logger: Logger):
        super().__init__(config, logger)
        if (self.fcm_concept_count % 2) != 0:
            raise ValueError(
                f"Concept count for symmetric one vs one classifier must be even, got {self.fcm_concept_count}")

    def fit(self, dataset: SeriesDataset) -> None:
        binary_classifiers: List[BinaryClassifierItem] = []

        for class_idx_1 in range(dataset.n_classes):
            for class_idx_2 in range(class_idx_1 + 1, dataset.n_classes):
                truncated_dataset = dataset.truncate((class_idx_1, class_idx_2))
                cmeans_computer = CMeansComputer(self.config)
                centroids_1 = cmeans_computer.compute(np.vstack(truncated_dataset.get_series_list(0)),
                                                      self.fcm_concept_count // 2)
                centroids_2 = cmeans_computer.compute(np.vstack(truncated_dataset.get_series_list(1)),
                                                      self.fcm_concept_count // 2)
                centroids = np.vstack([centroids_1, centroids_2])
                cmeans_transformer = CMeansTransformer(self.config, centroids)
                assert cmeans_transformer.num_centroids == self.fcm_concept_count

                binary_classifier = BinaryClassifier(self.config, cmeans_transformer, self.logger)
                binary_classifier.fit(truncated_dataset)
                assert binary_classifier.is_fitted

                binary_classifiers.append(
                    BinaryClassifierItem(model=binary_classifier, reference_classes=(class_idx_1, class_idx_2)))

        assert 2 * len(binary_classifiers) == dataset.n_classes * (dataset.n_classes - 1)
        self.binary_classifiers = tuple(binary_classifiers)


class CombinedOneVsOneClassifier(EnsembleClassifier):
    def fit(self, dataset: SeriesDataset) -> None:
        binary_classifiers: List[BinaryClassifierItem] = []

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

                binary_classifiers.append(
                    BinaryClassifierItem(model=binary_classifier, reference_classes=(class_idx_1, class_idx_2)))

        assert 2 * len(binary_classifiers) == dataset.n_classes * (dataset.n_classes - 1)
        self.binary_classifiers = tuple(binary_classifiers)
