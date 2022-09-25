from typing import NamedTuple, List, Tuple, Iterable, TypeVar, Callable

import numpy as np

DatasetItem = NamedTuple('DatasetItem', [('label', str), ('series_list', Tuple[np.ndarray])])

T = TypeVar('T')


class SeriesDataset:

    def truncate(self, indices_to_leave: List[int]) -> "SeriesDataset":
        return SeriesDataset([self._items[idx] for idx in indices_to_leave])

    def transform(self, mapping: Callable[[np.ndarray], T]) -> Tuple[Tuple[T]]:
        return ((mapping(x) for x in item.series_list) for item in self._items)

    def __init__(self, items: List[DatasetItem] = []):
        self._items: List[DatasetItem] = items

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


