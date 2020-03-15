import statistics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import plot_functions as plot
from classification_model import Classifier
import evaluation
import preprocessing as prep


class RandomForest(Classifier):

    def __init__(self):
        self.param_grid = {'num_estimators': [5, 7, 10, 17, 25, 30, 50, 75, 100, 150, 200, 300],
                           'max_depths': [5, 10, 15, 25, 50],
                           'max_features': ['sqrt', 'log2'],
                           'num_folds': 5}

        self.best_solution = {'n_estimators': 0,
                              'max_depths': 0,
                              'max_features': 'noname',
                              'accuracy': 0,
                              'sensitivity': 0,
                              'confusion_matrix': [0, 0, 0, 0]}

    def random_forest(self, data, labels, num_features, num_folds):

        fig_acc, axs_acc = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
        fig_sens, axs_sens = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
        if num_folds:
            self.param_grid['num_folds'] = num_folds

        skf = StratifiedKFold(n_splits=self.param_grid.get('num_folds'))

        for max_features_index in range(len(self.param_grid.get('max_features'))):
            max_features = self.param_grid.get('max_features')[max_features_index]
            values_acc = {}
            values_sens = {}
            for depth in self.param_grid.get('max_depths'):
                accuracies_values = []
                sensitivities_values = []
                for num_estimators in self.param_grid.get('num_estimators'):
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
                        rf = RandomForestClassifier(n_estimators=num_estimators, max_depth=depth, max_features=max_features)
                        rf.fit(trn_x, trn_y)
                        prd_y = rf.predict(tst_x)

                        # EVALUATION
                        evaluation.append_solution_for_fold(fold_accuracies,
                                                            fold_sensitivities,
                                                            fold_predictions,
                                                            fold,
                                                            tst_y,
                                                            prd_y)
                        fold = fold + 1

                    if statistics.mean(fold_accuracies) > self.best_solution.get('accuracy'):
                        self.best_solution['num_estimators'] = max_features_index
                        self.best_solution['max_depths'] = depth
                        self.best_solution['max_features'] = num_estimators
                        self.best_solution['accuracy'] = statistics.mean(fold_accuracies)
                        self.best_solution['sensitivity'] = statistics.mean(fold_sensitivities)

                        TN, FP, FN, TP = evaluation.compute_confm_values(fold_predictions)
                        self.best_solution['confusion_matrix'] = np.array(([TN, FP], [FN, TP]))

                    accuracies_values.append(statistics.mean(fold_accuracies))
                    sensitivities_values.append(statistics.mean(fold_sensitivities))

                values_acc[depth] = accuracies_values
                values_sens[depth] = sensitivities_values
            plot.multiple_line_chart(axs_acc[0, max_features_index], self.param_grid.get('num_estimators'), values_acc,
                                     'Random Forests with %s features' % max_features,
                                     'max_depths',
                                     'nr estimators',
                                     'accuracy',
                                     percentage=True)
            plt.figure()
            plot.multiple_line_chart(axs_sens[0, max_features_index], self.param_grid.get('num_estimators'),
                                     values_sens,
                                     'Random Forests with %s features' % max_features,
                                     'max_depths',
                                     'nr estimators',
                                     'sensitivity',
                                     percentage=True)
        plt.show()
        fig, axs = plt.subplots(1, 1, figsize=(4, 4), squeeze=False)
        plot.plot_confusion_matrix(axs[0, 0], self.best_solution.get('confusion_matrix'), 'Confusion matrix', [0, 1],
                                   True)
        plt.show()
        return self.best_solution
