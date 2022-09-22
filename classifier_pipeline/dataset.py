from typing import NamedTuple

DatasetItem = NamedTuple('DatasetItem', [('label', str), ('series_list', List[np.ndarray])])


class SeriesDataset:
    def __init__(self):
        self.items: List[DatasetItem] = []

    def add_item(self, label: str, series_list: List[np.ndarray]):
        self.items.append(DatasetItem(label=label, series_list=series_list))

    def get_series_list(self, idx: int) -> List[np.ndarray]:
        return self.items[idx].series_list

    def get_label(self, idx: int) -> str:
        return self.items[idx].label
