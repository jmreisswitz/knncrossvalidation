import numpy as np


class EvaluationService:
    @staticmethod
    def accuracy_score(test_predictions: np.array, test_labels: np.array) -> float:
        correct_predictions = 0
        for i in range(len(test_predictions)):
            if test_predictions[i] == test_labels[i]:
                correct_predictions += 1
        return correct_predictions/len(test_predictions)

    @staticmethod
    def precisions_score(test_predictions: np.array, test_labels: np.array) -> float:
        true_positives = EvaluationService._get_true_positives(test_predictions, test_labels)
        false_positives = EvaluationService._get_false_positives(test_predictions, test_labels)
        return true_positives/(true_positives + false_positives)

    @staticmethod
    def recall_score(test_predictions: np.array, test_labels: np.array) -> float:
        true_positives = EvaluationService._get_true_positives(test_predictions, test_labels)
        false_negatives = EvaluationService._get_false_negatives(test_predictions, test_labels)
        return true_positives / (true_positives + false_negatives)

    @staticmethod
    def f_score(test_predictions: np.array, test_labels: np.array, beta=1.0) -> float:
        beta_square = beta*beta
        precision = EvaluationService.precisions_score(test_predictions, test_labels)
        recall = EvaluationService.recall_score(test_predictions, test_labels)
        return (1 + beta_square)*(precision * recall)/((beta_square * precision) + recall)

    @staticmethod
    def _get_true_positives(test_predictions, test_labels) -> int:
        true_positives = 0
        for i in range(len(test_predictions)):
            if test_predictions[i] == test_labels[i] and test_predictions[i] == 1:
                true_positives += 1
        return true_positives

    @staticmethod
    def _get_false_positives(test_predictions, test_labels) -> int:
        false_positives = 0
        for i in range(len(test_predictions)):
            if test_predictions[i] != test_labels[i] and test_predictions[i] == 1:
                false_positives += 1
        return false_positives

    @staticmethod
    def _get_false_negatives(test_predictions, test_labels) -> int:
        false_negatives = 0
        for i in range(len(test_predictions)):
            if test_predictions[i] != test_labels[i] and test_predictions[i] == 0:
                false_negatives += 1
        return false_negatives
