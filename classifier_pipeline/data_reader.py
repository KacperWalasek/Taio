import os
from typing import NamedTuple, List

import numpy as np

from classifier_pipeline.dataset import SeriesDataset

ClassDir = NamedTuple("ClassDir", [('name', str), ('path', str)])


class DataReader:
    TRAIN_SUBDIR = "Train"
    TEST_SUBDIR = "Test"

    def read_preprocess_train(self, root_dir: str, dataset_name: str) -> SeriesDataset:
        return self._read_preprocess(root_dir, dataset_name, self.TRAIN_SUBDIR)

    def read_preprocess_test(self, root_dir: str, dataset_name: str) -> SeriesDataset:
        return self._read_preprocess(root_dir, dataset_name, self.TEST_SUBDIR)

    def _read_preprocess(self, root_dir: str, dataset_name: str, subdir: str) -> SeriesDataset:
        class_dirs = [ClassDir(name=entry.name, path=entry.path) for entry in
                      os.scandir(f"{root_dir}/{dataset_name}/{subdir}")
                      if entry.is_dir()]
        ret = SeriesDataset()
        for class_dir in class_dirs:
            series_list: List[np.ndarray] = []
            for file in os.scandir(class_dir.path):
                series_array = self._read_preprocess_single_series(f"{class_dir.path}/{file}")
                series_list.append(series_array)
            ret.add_item(class_dir.name, series_list)
        return ret

    def _read_preprocess_single_series(self, filepath: str) -> np.ndarray:
        """

        @param filepath: a path to a series stored in csv format
        @return: ndarray of shape (num_observations, 2*dimension)
        """
        series_array = np.genfromtxt(filepath, delimiter=",", ndmin=2)
        series_array = self._remove_nans(series_array)
        series_array = self._remove_redundant_rows(series_array)
        series_array = self._add_adjacent_differences(series_array)
        return series_array

    @staticmethod
    def _remove_nans(series_array: np.ndarray) -> np.ndarray:
        return series_array[~np.isnan(series_array).any(axis=1)]

    @staticmethod
    def _remove_redundant_rows(series_array: np.ndarray):
        first = series_array[0]
        i = 1
        while i < series_array.shape[0] and np.array_equal(first, series_array[i]):
            i = i + 1
        series_array = series_array[i - 1:]

        last = series_array[-1]
        i = 2
        while i <= series_array.shape[0] and np.array_equal(last, series_array[-i]):
            i = i + 1
        return series_array if i == 2 else series_array[: -(i - 2)]

    @staticmethod
    def _add_adjacent_differences(series_array: np.ndarray) -> np.ndarray:
        return np.hstack([
            series_array[1:],
            np.diff(series_array, axis=0)
        ])
