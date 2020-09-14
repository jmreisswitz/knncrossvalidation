from typing import Tuple

import numpy as np
import pandas as pd

from k_fold import Kfold
from knn import Knn
from trainer import Train
from evaluation_service import EvaluationService


class KnnKFoldTrain(Train):
    def __init__(self, dataset: pd.DataFrame, label_column: str):
        super().__init__(dataset, label_column)
        labels = np.array(self.dataset[self.label_column])
        features = np.array(self.dataset.drop(self.label_column, axis=1))
        self.k_folds = Kfold(features, labels, folds_num=10)

    def _process_dataset(self, original_dataset: pd.DataFrame) -> pd.DataFrame:
        for column in original_dataset.columns:
            max_ = max(original_dataset[column])
            min_ = min(original_dataset[column])
            original_dataset[column] = original_dataset[column].apply(lambda x: (x-min_)/(max_ - min_))
        return original_dataset

    def _separate_dataset(self) -> Tuple[np.array, np.array, np.array, np.array]:
        # train_features, test_features, train_labels, test_labels
        return self.k_folds.get_next_fold()

    def _get_model(self):
        return Knn()

    def _evaluate_model(self, test_predictions, test_labels):
        return EvaluationService.accuracy_score(test_predictions, test_labels), EvaluationService.f_score(test_predictions, test_labels)


def get_dataset():
    return pd.read_csv('diabetes.csv')


def main():
    knn_train = KnnKFoldTrain(get_dataset(), 'Outcome')
    print('fold,acc,f1')
    for i in range(knn_train.k_folds.folds_num):
        prediction = knn_train.execute()
        print(f'{i},{prediction[0]},{prediction[1]}')


if __name__ == '__main__':
    main()
