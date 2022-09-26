from abc import ABC
from collections.abc import MutableMapping
from logging import Logger
from typing import Literal, Tuple, List

import numpy as np

from classifier_pipeline.binary_classifier import BinaryClassifier
from classifier_pipeline.fuzzy_cmeans import CMeansComputer, CMeansTransformer
from classifier_pipeline.dataset import SeriesDataset


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

    def fit(self, dataset: SeriesDataset) -> None:
        pass


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
            centroids = cmeans_computer.compute(np.vstack(idx_vs_all_dataset.get_series_list([0])))
            cmeans_transformer = CMeansTransformer(self.config, centroids)
            assert cmeans_transformer.num_centroids == self.fcm_concept_count

            binary_classifier = BinaryClassifier(self.config, cmeans_transformer, self.logger)
            binary_classifier.fit(idx_vs_all_dataset)
            assert binary_classifier.is_fitted

            binary_classifiers.append(binary_classifier)

        self.binary_classifiers = tuple(binary_classifiers)

    def evaluate(self, dataset: SeriesDataset) -> float:
        pass


class AsymmetricOneVsOneClassifier(EnsembleClassifier):
    pass


class SymmetricOneVsOneClassifier(EnsembleClassifier):
    pass


class CombinedOneVsOneClassifier(EnsembleClassifier):
    pass
