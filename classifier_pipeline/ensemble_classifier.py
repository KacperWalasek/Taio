from collections.abc import MutableMapping
from typing import Literal

from classifier_pipeline.dataset import SeriesDataset


class EnsembleClassifier:
    def __init__(self, method: Literal['1_vs_all', 'asymmetric_1_vs_1', 'symmetric_1_vs_1',
                                                 'combined_symmetric_1_vs_1'], config: MutableMapping):
        pass

    def fit(self, dataset: SeriesDataset) -> None:



class OneVsAllClassifier(EnsembleClassifier):
    pass

class AsymmetricOneVsOneClassifier(EnsembleClassifier):
    pass

class SymmetricOneVsOneClassifier(EnsembleClassifier):
    pass

class CombinedOneVsOneClassifier(EnsembleClassifier):
    pass