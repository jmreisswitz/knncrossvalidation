from abc import abstractmethod, ABC
from typing import Tuple
import pandas as pd
import numpy as np


class Train(ABC):
    def __init__(self, dataset: pd.DataFrame, label_column: str):
        self.label_column = label_column
        self.dataset = self._process_dataset(dataset)
        self.model = self._get_model()

    def execute(self):
        train_features, test_features, train_labels, test_labels = self._separate_dataset()
        self._train(train_features, train_labels)
        test_predictions = self._predict(test_features)
        return self._evaluate_model(test_predictions, test_labels)

    def _train(self, train_features: np.array, train_labels: np.array) -> None:
        self.model.fit(train_features, train_labels)

    def _predict(self, test_features: np.array) -> np.array:
        return self.model.predict(test_features)

    @abstractmethod
    def _process_dataset(self, original_dataset):
        pass

    @abstractmethod
    def _separate_dataset(self) -> Tuple[np.array, np.array, np.array, np.array]:
        pass

    @abstractmethod
    def _get_model(self):
        pass

    @abstractmethod
    def _evaluate_model(self, test_predictions, test_labels):
        pass


