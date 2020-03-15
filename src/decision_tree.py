import statistics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

import plot_functions as plot
from classification_model import Classifier
import evaluation
import preprocessing as prep


class DecisionTree(Classifier):

    def __init__(self):
        self.param_grid = {'min_samples_leaf': [.05, .025, .01, .0075, .005, .0025, .001],
                           'max_depths': [5, 10, 25, 50],
                           'criteria': ['entropy', 'gini'],
                           'num_folds': 5}

        self.best_solution = {'criteria': 'no_name',
                              'min_samples_leaf': 0,
                              'max_depths': 0,
                              'accuracy': 0,
                              'sensitivity': 0,
                              'confusion_matrix': [0, 0, 0, 0]}

    def decision_tree(self, data, labels, num_features, num_folds=5):
        fig_acc, axs_acc = plt.subplots(1, 2, figsize=(13, 4), squeeze=False)
        fig_sens, axs_sens = plt.subplots(1, 2, figsize=(13, 4), squeeze=False)

        if num_folds:
            self.param_grid['num_folds'] = num_folds
        skf = StratifiedKFold(n_splits=self.param_grid.get('fold'))

        for criteria_index in range(len(self.param_grid.get('criteria'))):
            criteria = self.param_grid.get('criteria')[criteria_index]
            values_acc = {}
            values_sens = {}
            for depth in self.param_grid.get('max_depths'):
                accuracies_values = []
                sensitivities_values = []
                for num_samples in self.param_grid.get('min_samples_leaf'):
                    tree = DecisionTreeClassifier(min_samples_leaf=num_samples, max_depth=depth, criterion=criteria,
                                                  min_impurity_decrease=0.005)
                    fold_accuracies, fold_sensitivities, fold_predictions, fold = evaluation.initialize_metrics(self)

                    # there are four probes for every patient, it's reasonable to take only one
                    single_data, single_labels = prep.select_single_probes(data, labels)

                    for train_index, test_index in skf.split(single_data, single_labels):
                        trn_x, tst_x, trn_y, tst_y = prep.separate_and_prepare_data(data,
                                                                                    labels,
                                                                                    train_index,
                                                                                    test_index,
                                                                                    num_features)

                        # CLASSIFICATION
                        tree.fit(trn_x, trn_y)
                        prd_y = tree.predict(tst_x)

                        # EVALUATION
                        evaluation.append_solution_for_fold(fold_accuracies,
                                                            fold_sensitivities,
                                                            fold_predictions,
                                                            fold,
                                                            tst_y,
                                                            prd_y)
                        fold = fold + 1

                    if statistics.mean(fold_accuracies) > self.best_solution.get('accuracy'):
                        self.best_solution['criteria'] = self.param_grid.get('criteria')[criteria_index]
                        self.best_solution['min_samples_leaf'] = num_samples
                        self.best_solution['max_depths'] = depth
                        self.best_solution['accuracy'] = statistics.mean(fold_accuracies)
                        self.best_solution['sensitivity'] = statistics.mean(fold_sensitivities)

                        TN, FP, FN, TP = evaluation.compute_confm_values(fold_predictions)
                        self.best_solution['confusion_matrix'] = np.array(([TN, FP], [FN, TP]))
                    accuracies_values.append(statistics.mean(fold_accuracies))
                    sensitivities_values.append(statistics.mean(fold_sensitivities))

                values_acc[depth] = accuracies_values
                values_sens[depth] = sensitivities_values
            plot.multiple_line_chart(axs_acc[0, criteria_index], self.param_grid.get('min_samples_leaf'), values_acc,
                                     'Decision Trees with %s criteria' % criteria,
                                     'max_depths',
                                     'min_samples_leaf',
                                     'accuracy',
                                     percentage=True)

            plot.multiple_line_chart(axs_sens[0, criteria_index], self.param_grid.get('min_samples_leaf'), values_sens,
                                     'Decision Trees with %s criteria' % criteria,
                                     'max_depths',
                                     'min_samples_leaf',
                                     'sensitivity',
                                     percentage=True)
        plt.show()
        fig, axs = plt.subplots(1, 1, figsize=(4, 4), squeeze=False)
        plot.plot_confusion_matrix(axs[0, 0], self.best_solution.get('confusion_matrix'), 'Confusion matrix', [0, 1],
                                   True)
        plt.show()
        return self.best_solution
