import random
from logging import Logger
from math import ceil
from typing import NamedTuple, List, Tuple, Iterable, TypeVar, Callable

import numpy as np

DatasetItem = NamedTuple('DatasetItem', [('label', str), ('series_list', Tuple[np.ndarray, ...])])

T = TypeVar('T')


class SeriesDataset:
    ALL_OTHER_LABEL = "ALL_OTHERS"

    def select_classes(self, indices_to_leave: Tuple[int, ...]) -> "SeriesDataset":
        """
        Creates a new dataset with only selected class indices
        @param indices_to_leave:
        @return: a new dataset with classes specified by indices
        """
        return SeriesDataset([self._items[idx] for idx in indices_to_leave])

    def truncate(self, length_fraction: float, min_length: int = None, logger: Logger = None) -> "SeriesDataset":
        """
        Creates a new dataset with truncated series for early classification task
        @param length_fraction: a number in [0, 1] interval specifying desired length fraction of new series
        @param min_length: minimum size of a series to include it in the dataset
        @param logger: logger for logging the number of entries which do not meet min_size requirement
        @return: a new dataset with truncated series
        """
        new_items = [DatasetItem(label=item.label, series_list=tuple(x[:new_length] for x in item.series_list if (
            new_length := ceil(x.shape[0] * length_fraction)) >= min_length)) for item in self._items]
        num_too_short = sum(len(x.series_list) for x in self._items) - sum(len(x.series_list) for x in new_items)
        assert num_too_short >= 0
        if num_too_short and logger:
            logger.warning(f"Number of too short series removed during truncation: {num_too_short}")
        return SeriesDataset(new_items)

    def make_one_vs_all(self, reference_class_idx: int) -> "SeriesDataset":
        """
        Creates a new dataset two-class dataset consisting of the reference class and all other classes merged.
        Both classes are of similar sizes
        @param reference_class_idx:
        @return: a new dataset for two-class classification task
        """
        other_series_list = []
        fraction_to_take = 1. / (len(self._items) - 1)
        for i, item in enumerate(self._items):
            if i != reference_class_idx:
                num_series_to_take = ceil(fraction_to_take * len(item.series_list))
                other_series_list.extend(random.sample(item.series_list, num_series_to_take))
        new_items = [self._items[reference_class_idx],
                     DatasetItem(label=self.ALL_OTHER_LABEL, series_list=tuple(other_series_list))]
        return SeriesDataset(new_items)

    def transform(self, mapping: Callable[[np.ndarray], T]) -> Tuple[List[T], ...]:
        """
        Transforms all series
        @param mapping: mapping to transform single series
        @return: tuple of transformed series lists, each tuple item corresponds to a single class
        """
        return tuple([mapping(x) for x in item.series_list] for item in self._items)

    def __init__(self, items: List[DatasetItem] = None):
        """
        @param items: optional list of items to include in the dataset
        """
        self._items: List[DatasetItem] = items or []

    @property
    def n_classes(self) -> int:
        """
        @return: a number of all classes
        """
        return len(self._items)

    @property
    def labels(self) -> List[str]:
        """
        @return: a list of all labels
        """
        return [x.label for x in self._items]

    def add_class_item(self, label: str, series_list: Iterable[np.ndarray]) -> None:
        """
        Adds a new class entry
        @param label: a class label
        @param series_list: a list of series for the added class
        """
        if label in (x.label for x in self._items):
            raise ValueError(f"Class with label {label} already in the datasets")
        self._items.append(DatasetItem(label=label, series_list=tuple(series_list)))

    def get_series_list(self, idx: int) -> List[np.ndarray]:
        """
        @param idx: a class index
        @return: a list of series for a given class index
        """
        return self._items[idx].series_list

    def get_label(self, idx: int) -> str:
        """
        @param idx: a class index
        @return: a label for a given class index
        """
        return self._items[idx].label
