import logging
from collections import defaultdict

import numpy as np
from scipy.spatial import distance


class Knn:
    def __init__(self, k_neighbours=5):
        self.k_neighbours = k_neighbours
        self.features = None
        self.labels = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def fit(self, train_features: np.array, train_labels: np.array):
        self.features = train_features
        self.labels = train_labels

    def predict(self, test_features) -> np.array:
        test_predictions = []
        for test_feature in test_features:
            distances = self._calculate_distance_from_dataset(test_feature)
            ordered_results = self._order_results(distances)
            test_predictions.append(self._get_prediction(ordered_results))
        return np.array(test_predictions)

    def _calculate_distance_from_dataset(self, test_feature):
        distances = []
        row_index = 0
        for dataset_feature in self.features:
            distance_from_test_point = distance.euclidean(test_feature, dataset_feature)
            distances.append((row_index, distance_from_test_point))
            row_index += 1
        self.logger.debug(f'Distances : \n{distances[:3]}')
        return distances

    @staticmethod
    def _order_results(distances: np.array):
        return sorted(distances, key=lambda x: x[1])  # 0 is index

    def _get_prediction(self, ordered_results: np.array):
        results_dict = defaultdict(int)
        for index, value in ordered_results[:self.k_neighbours]:
            results_dict[self.labels[index]] += 1
        self.logger.debug(f'Test predictions: \n{results_dict}')
        return max(results_dict.items(), key=lambda x: x[1])[0]
