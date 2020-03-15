import statistics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

import plot_functions as plot
from classification_model import Classifier
import evaluation
import preprocessing as prep


class InstanceBased(Classifier):

    def __init__(self):
        self.param_grid = {'dist': ['manhattan', 'euclidean', 'chebyshev'],
                           'n_neighbors': np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 25, 30, 50, 70]),
                           'num_folds': 5}

        self.best_solution = {'dist': 'no_name',
                              'n_neighbors': 0,
                              'accuracy': 0,
                              'sensitivity': 0,
                              'confusion_matrix': [0, 0, 0, 0]}

    def knn(self, data, labels, num_features, num_folds=5):

        fig_acc, axs_acc = plt.subplots(1, 1, figsize=(6, 4), squeeze=False)
        fig_sens, axs_sens = plt.subplots(1, 1, figsize=(6, 4), squeeze=False)
        if num_folds:
            self.param_grid['num_folds'] = num_folds
        skf = StratifiedKFold(n_splits=self.param_grid.get('num_folds'))

        values_acc = {}
        values_sens = {}
        for dist_index, dist in enumerate(self.param_grid.get('dist')):
            accuracies_values = []
            sensitivities_values = []
            for n_neigh in self.param_grid.get('n_neighbors'):
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
                    knn = KNeighborsClassifier(n_neighbors=n_neigh, metric=dist)
                    knn.fit(trn_x, trn_y)
                    prd_y = knn.predict(tst_x)

                    # EVALUATION
                    evaluation.append_solution_for_fold(fold_accuracies,
                                                        fold_sensitivities,
                                                        fold_predictions,
                                                        fold,
                                                        tst_y,
                                                        prd_y)
                    fold = fold + 1

                if statistics.mean(fold_accuracies) > self.best_solution.get('accuracy'):
                    self.best_solution['dist'] = self.param_grid.get('dist')[dist_index]
                    self.best_solution['n_neighbors'] = n_neigh
                    self.best_solution['accuracy'] = statistics.mean(fold_accuracies)
                    self.best_solution['sensitivity'] = statistics.mean(fold_sensitivities)

                    TN, FP, FN, TP = evaluation.compute_confm_values(fold_predictions)
                    self.best_solution['confusion_matrix'] = np.array(([TN, FP], [FN, TP]))
                # result for different number of neighbours
                accuracies_values.append(statistics.mean(fold_accuracies))
                sensitivities_values.append(statistics.mean(fold_sensitivities))
            # results for every distance with different num of neighbours
            values_acc[dist] = accuracies_values
            values_sens[dist] = sensitivities_values
        plot.multiple_line_chart(axs_acc[0, 0], self.param_grid.get('n_neighbors'), values_acc,
                                 'KNN for different number of neighbours',
                                 'Distance metrics',
                                 'nr neighbours',
                                 'accuracy',
                                 percentage=False)
        plot.multiple_line_chart(axs_sens[0, 0], self.param_grid.get('n_neighbors'), values_sens,
                                 'KNN for different number of neighbours',
                                 'Distance metrics',
                                 'nr neighbours',
                                 'sensitivity',
                                 percentage=False)
        plt.show()
        fig, axs = plt.subplots(1, 1, figsize=(4, 4), squeeze=False)
        plot.plot_confusion_matrix(axs[0, 0], self.best_solution.get('confusion_matrix'), 'Confusion matrix', [0, 1],
                                   True)
        plt.show()
        return self.best_solution
