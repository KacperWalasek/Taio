import configparser
import unittest

import numpy as np

from classifier_pipeline.fuzzy_cmeans import CMeansComputer, CMeansTransformer


class TestCMeansComputer(unittest.TestCase):
    def test_compute(self):
        points = np.vstack([np.array([[100, 150, 200]] * 100), np.array([[-1000, -1000, -500]] * 100)])
        np.random.shuffle(points)
        config = configparser.ConfigParser()
        config["FuzzyCMeans"] = {"M": 2, "Error": 1e-8, "Maxiter": 1e6}
        cmeans_computer = CMeansComputer(config)
        ret = cmeans_computer.compute(points, 2)
        self.assertEqual((2, 3), ret.shape)
        self.assertTrue((np.allclose([100, 150, 200], ret[0]) and np.allclose([-1000, -1000, -500], ret[1])) or (
                np.allclose([100, 150, 200], ret[1]) and np.allclose([-1000, -1000, -500], ret[0])))


class TestCMeansTransformer(unittest.TestCase):
    def test_transform(self):
        centroids = np.array([[1000, 2000, 500], [-1000, -2000, -500]])
        config = configparser.ConfigParser()
        config["FuzzyCMeans"] = {"M": 2, "Error": 1e-8, "Maxiter": 1e6}
        cmeans_transformer = CMeansTransformer(config, centroids)
        self.assertEqual(2, cmeans_transformer.num_centroids)
        expected = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        ret = cmeans_transformer.transform(
            np.array([[1000.1, 2000.2, 500.1], [-1000.05, -2000.3, -500.1], [0, 0, 0], [1000, 2000, 500]]))
        self.assertTrue(np.allclose(expected, ret))
