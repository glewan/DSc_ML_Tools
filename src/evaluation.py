import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def initialize_metrics(classifier):
    fold_accuracies = []
    fold_sensitivities = []
    fold_predictions = np.empty((classifier.param_grid.get('fold_num'), 4))
    fold = 0

    return fold_accuracies, fold_sensitivities, fold_predictions, fold


def compute_confm_values(fold_predictions):
    TN = sum(fold_predictions[:, 0])
    FP = sum(fold_predictions[:, 1])
    FN = sum(fold_predictions[:, 2])
    TP = sum(fold_predictions[:, 3])

    return TN, FP, FN, TP


def evaluate_fold_solution(tst_y, prd_y):
    fold_accuracy = metrics.accuracy_score(tst_y, prd_y)
    fold_sensitivity = metrics.recall_score(tst_y, prd_y)
    tn, fp, fn, tp = confusion_matrix(tst_y, prd_y).ravel()
    return fold_accuracy, fold_sensitivity, [tn, fp, fn, tp]


def append_solution_for_fold(fold_accuracies, fold_sensitivities, fold_predictions, fold, tst_y, prd_y):
    acc, sns, cm = evaluate_fold_solution(tst_y, prd_y)
    fold_accuracies.append(acc)
    fold_sensitivities.append(sns)
    fold_predictions[fold] = cm
