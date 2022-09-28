import random
from math import ceil
from typing import NamedTuple, List, Tuple, Iterable, TypeVar, Callable

import numpy as np

DatasetItem = NamedTuple('DatasetItem', [('label', str), ('series_list', Tuple[np.ndarray])])

T = TypeVar('T')


class SeriesDataset:
    ALL_OTHER_LABEL = "all_other"

    def truncate(self, indices_to_leave: Tuple[int]) -> "SeriesDataset":
        return SeriesDataset([self._items[idx] for idx in indices_to_leave])

    def transform(self, mapping: Callable[[np.ndarray], T]) -> Tuple[List[T]]:
        return tuple([mapping(x) for x in item.series_list] for item in self._items)

    def make_one_vs_all(self, reference_class_idx: int):
        other_series_list = []
        fraction_to_take = 1. / (len(self._items) - 1)
        for i, item in enumerate(self._items):
            if i != reference_class_idx:
                num_series_to_take = ceil(fraction_to_take * len(item.series_list))
                other_series_list.extend(random.sample(item.series_list, num_series_to_take))
        new_items = [self._items[reference_class_idx],
                     DatasetItem(label=self.ALL_OTHER_LABEL, series_list=tuple(other_series_list))]
        return SeriesDataset(new_items)

    def __init__(self, items: List[DatasetItem] = None):
        self._items: List[DatasetItem] = items or []

    @property
    def n_classes(self) -> int:
        return len(self._items)

    @property
    def labels(self) -> List[str]:
        return [x.label for x in self._items]

    def add_class_item(self, label: str, series_list: Iterable[np.ndarray]):
        if label in (x.label for x in self._items):
            raise ValueError(f"Class with label {label} already in the datasets")
        self._items.append(DatasetItem(label=label, series_list=tuple(series_list)))

    def get_series_list(self, idx: int) -> List[np.ndarray]:
        return self._items[idx].series_list

    def get_label(self, idx: int) -> str:
        return self._items[idx].label
