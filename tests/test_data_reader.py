import tempfile
import unittest

import numpy as np

from classifier_pipeline.data_reader import DataReader


class TestDataReader(unittest.TestCase):
    def test_preprocess_single_series_1d(self):
        with tempfile.TemporaryDirectory() as dir:
            with open(f"{dir}/series.csv", "w") as f:
                f.writelines(f"{s}\n" for s in
                             ["2.5e+00", "2.5e+00", "2.5e+00", "9.0e-01", "5.0e+00", "5.0e+00", "6e+00", "6e+00"])
            data_reader = DataReader("trash", "trash")
            expected = np.array([[0.9, 0.9 - 2.5], [5, 5 - 0.9], [5, 0], [6, 1]])
            ret = data_reader._read_preprocess_single_series(f"{dir}/series.csv")
            self.assertTrue(np.allclose(expected, ret))

    def test_preprocess_single_series_2d(self):
        with tempfile.TemporaryDirectory() as dir:
            with open(f"{dir}/series.csv", "w") as f:
                f.writelines(f"{s}\n" for s in
                             ["2.5e+00,5", "2.5e+00,5", "2.5e+00,6", "9.0e-01,6", "5.0e+00,-2", "5.0e+00,3", "6e+00,4",
                              "6e+00,4"])
            data_reader = DataReader("trash", "trash")
            expected = np.array(
                [[2.5, 6, 0, 1], [0.9, 6, 0.9 - 2.5, 0], [5, -2, 5 - 0.9, -8], [5, 3, 0, 5], [6, 4, 1, 1]])
            ret = data_reader._read_preprocess_single_series(f"{dir}/series.csv")
            self.assertTrue(np.allclose(expected, ret))
