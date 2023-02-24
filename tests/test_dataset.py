import tempfile
import unittest

import numpy as np

from classifier_pipeline.dataset import DatasetItem, SeriesDataset


class TestDataReader(unittest.TestCase):
    def test_split_one_class(self):
        dataset = SeriesDataset(
            [DatasetItem(str(x), (np.array([1]), np.array([2]))) for x in range(1)]
        )
        self.assertEqual(2, dataset.n_series)
        self.assertEqual(1, dataset.n_classes)
        ds_1, ds_2 = dataset.split(0.5)
        self.assertEqual(1, ds_1.n_series)
        self.assertEqual(1, ds_2.n_series)

    def test_split_at_least_one_from_each_class(self):
        dataset = SeriesDataset(
            [DatasetItem(str(x), (np.array([1]), np.array([2]))) for x in range(4)]
        )
        self.assertEqual(8, dataset.n_series)
        self.assertEqual(4, dataset.n_classes)
        ds_1, ds_2 = dataset.split(1.0)
        self.assertEqual(dataset.n_classes, ds_2.n_classes)
        for idx in range(dataset.n_classes):
            self.assertEqual(1, len(ds_1.get_series_list(idx)))
            self.assertEqual(1, len(ds_2.get_series_list(idx)))
            self.assertNotEqual(ds_1.get_series_list(idx)[0], ds_2.get_series_list(idx)[0])

    def test_split_max_ratio(self):
        dataset = SeriesDataset(
            [DatasetItem(str(x), (np.array([1]), np.array([2]), np.array([3]))) for x in range(2)]
        )
        self.assertEqual(6, dataset.n_series)
        self.assertEqual(2, dataset.n_classes)
        ds_1, ds_2 = dataset.split(0.5)
        self.assertEqual(dataset.n_classes, ds_2.n_classes)
        for idx in range(dataset.n_classes):
            self.assertEqual(1, len(ds_1.get_series_list(idx)))
            self.assertEqual(2, len(ds_2.get_series_list(idx)))

    def test_split_(self):
        dataset = SeriesDataset(
            [DatasetItem(str(x), tuple(np.array([x]) for x in range(10))) for x in range(2)]
        )
        self.assertEqual(20, dataset.n_series)
        self.assertEqual(2, dataset.n_classes)
        ds_1, ds_2 = dataset.split()
        self.assertEqual(dataset.n_classes, ds_2.n_classes)
        for idx in range(dataset.n_classes):
            self.assertEqual(8, len(ds_1.get_series_list(idx)))
            self.assertEqual(2, len(ds_2.get_series_list(idx)))
