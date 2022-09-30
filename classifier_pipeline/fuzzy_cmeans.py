import configparser

import numpy as np
import skfuzzy as fuzz


class CMeansComputer:
    """
    Class for calculating c-means centroids
    """

    def __init__(self, config: configparser.ConfigParser):
        """
        @param config: config to use
        """
        self.is_fitted = False
        self.cmeans_params = {
            "m": config.getint("FuzzyCMeans", "M"),
            "error": config.getfloat("FuzzyCMeans", "Error"),
            "maxiter": int(config.getfloat("FuzzyCMeans", "Maxiter"))
        }

    def compute(self, points: np.ndarray, num_clusters: int) -> np.ndarray:
        """
        Computes centroids
        @param points: ndarray of shape (num_points, dimension)
        @param num_clusters: number of clusters in the fuzzy space
        @return: fit centroids, ndarray of shape (num_centroids, dimension)
        """
        centroids = fuzz.cmeans(points.T, num_clusters, **self.cmeans_params)[0]
        return centroids


class CMeansTransformer:
    """
    Class for transforming series into fuzzy space
    """

    def __init__(self, config: configparser.ConfigParser, centroids: np.ndarray):
        """
        @param config: config to use
        @param centroids: ndarray of shape (num_centroids, dimension)
        """
        self.cmeans_params = {
            "m": config.getint("FuzzyCMeans", "M"),
            "error": config.getfloat("FuzzyCMeans", "Error"),
            "maxiter": int(config.getfloat("FuzzyCMeans", "Maxiter"))
        }
        self.centroids = centroids

    @property
    def num_centroids(self) -> int:
        """
        @return: number of centroids
        """
        return self.centroids.shape[0]

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Transform samples into fuzzy space (membership degrees to found centroids)
        @param points: ndarray of shape (num_points, dimension)
        @return: ndarray of shape (num_points, num_clusters)
        """
        return fuzz.cmeans_predict(points.T, self.centroids, **self.cmeans_params)[0].T
