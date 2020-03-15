import statistics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB

import plot_functions as plot
from classification_model import Classifier
import evaluation
import preprocessing as prep


class NaiveBayes(Classifier):

    def __init__(self):
        self.param_grid = {'estimators': {'GaussianNB': GaussianNB()},
                           'num_folds': 5}

        self.best_solution = {'estimator': 'no_name',
                              'accuracy': 0,
                              'sensitivity': 0,
                              'confusion_matrix': [0, 0, 0, 0]}

    def naive_bayes(self, data, labels, num_features, num_folds=5):
        if num_folds:
            self.param_grid['num_folds'] = num_folds

        skf = StratifiedKFold(n_splits=self.param_grid.get('num_folds'))

        for clf in self.param_grid.get('estimators'):
            fold_accuracies, fold_sensitivities, fold_predictions, fold = evaluation.initialize_metrics(self)

            # there are four probes for every patient, it's reasonable to take only one
            single_data, single_labels = prep.select_single_probes(data, labels)

            for train_index, test_index in skf.split(single_data, single_labels):
                trn_x, tst_x, trn_y, tst_y = self.separate_and_prepare_data(data,
                                                                            labels,
                                                                            train_index,
                                                                            test_index,
                                                                            num_features)
                # CLASSIFICATION
                self.param_grid.get('estimators')[clf].fit(trn_x, trn_y)
                prd_y = self.param_grid.get('estimators')[clf].predict(tst_x)

                # EVALUATION
                evaluation.append_solution_for_fold(fold_accuracies,
                                                    fold_sensitivities,
                                                    fold_predictions,
                                                    fold,
                                                    tst_y,
                                                    prd_y)
                fold = fold + 1

            # SAVE BEST SOLUTION
            if statistics.mean(fold_accuracies) > self.best_solution.get('accuracy'):
                self.best_solution['estimator'] = clf
                self.best_solution['accuracy'] = statistics.mean(fold_accuracies)
                self.best_solution['sensitivity'] = statistics.mean(fold_sensitivities)

                TN, FP, FN, TP = evaluation.compute_confm_values(fold_predictions)
                self.best_solution['confusion_matrix'] = np.array(([TN, FP], [FN, TP]))

        fig, axs = plt.subplots(1, 1, figsize=(4, 4), squeeze=False)
        plot.plot_confusion_matrix(axs[0, 0], self.best_solution.get('confusion_matrix'),
                                   'Confusion matrix',
                                   [0, 1], True)
        plt.show()
        return self.best_solution
