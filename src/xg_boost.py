import statistics
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

import plot_functions as plot
from classification_model import Classifier
import evaluation
import preprocessing as prep


class XGBoost(Classifier):

    def __init__(self):
        self.param_grid = {
            "learning_rate": [0.01],  # , 0.05],  # , 0.1, 0.15, 0.2, 0.3],
            "n_estimators": [5],  # , 7, 10, 17],  # , 25, 30, 50],
            "max_depth": [3],  # range(3, 10, 2),
            "min_child_weight": [2],  # range(1, 6, 2),
            'gamma': [0],  # [0, 2, 3, 5, 7, 10],
            "subsample": [0.7],  # [0.7, 0.8, 1, 1.5],
            "colsample_bytree": [0.7],  # [0.7, 0.8, 1, 1.5],
            "objective": 'binary:logistic',
            "num_folds": 5}

        self.best_solution = {'estimator': 'no_name',
                              'accuracy': 0,
                              'sensitivity': 0,
                              'confusion_matrix': [0, 0, 0, 0]}

    def select_params(self, data, labels, num_features):
        lr, n_est = self.lr_num_estimators_tuning(data, labels, num_features)
        self.best_solution['learning_rate'] = lr
        self.best_solution['n_estimators'] = n_est
        depth, weight = self.depth_min_child_weight_tuning(data, labels, num_features)
        self.best_solution['max_depth'] = depth
        self.best_solution['min_child_weight'] = weight
        gamma = self.gamma_tuning(data, labels, num_features)
        self.best_solution['gamma'] = gamma

    def xg_boost(self, data, labels, num_features, num_folds=5):
        if num_folds:
            self.param_grid['num_folds'] = num_folds

        self.select_params(data, labels, num_features)

        skf = StratifiedKFold(n_splits=self.param_grid.get('num_folds'))
        fold_accuracies, fold_sensitivities, fold_predictions, fold = evaluation.initialize_metrics(self)

        # there are four probes for every patient, it's reasonable to take only one
        single_data, single_labels = prep.select_single_probes(data, labels)

        for train_index, test_index in skf.split(single_data, single_labels):
            trn_x, tst_x, trn_y, tst_y = self.separate_and_prepare_data(data, labels, train_index, test_index, num_features)

            # CLASSIFICATION
            model = XGBClassifier(learning_rate=self.best_solution.get('learning_rate'),
                                  n_estimators=self.best_solution.get('n_estimators'),
                                  max_depth=self.best_solution.get('max_depth'),
                                  min_child_weight=self.best_solution.get('min_child_weight'),
                                  gamma=self.best_solution.get('gamma'),
                                  subsample=0.8,
                                  colsample_bytree=0.8,
                                  objective='binary:logistic',
                                  nthread=4,
                                  scale_pos_weight=1,
                                  seed=27)
            model.fit(trn_x, trn_y)
            prd_y = model.predict(tst_x)

            # EVALUATION
            evaluation.append_solution_for_fold(fold_accuracies,
                                                fold_sensitivities,
                                                fold_predictions,
                                                fold,
                                                tst_y,
                                                prd_y)
            fold = fold + 1
        TN, FP, FN, TP = evaluation.compute_confm_values(fold_predictions)

        self.best_solution['confusion_matrix'] = np.array(([TN, FP], [FN, TP]))
        self.best_solution['accuracy'] = statistics.mean(fold_accuracies)
        self.best_solution['sensitivity'] = statistics.mean(fold_sensitivities)
        return self.best_solution

    def lr_num_estimators_tuning(self, data, labels, num_features):
        best_solution = {}
        macro_accuracy = 0
        skf = StratifiedKFold(n_splits=self.param_grid.get('fold'))
        for lr in self.param_grid.get('learning_rate'):
            for n in self.param_grid.get('n_estimators'):
                fold_accuracies = []
                fold = 0
                single_indices = [i for i in range(0, len(labels), 3)]
                single_data = data.iloc[single_indices]
                single_labels = labels.iloc[single_indices]
                for train_index, test_index in skf.split(single_data, single_labels):
                    trnX, tstX, trnY, tstY = self.separate_and_prepare_data(data, labels, train_index, test_index, num_features)

                    # CLASSIFICATION
                    model = XGBClassifier(learning_rate =lr, n_estimators=n, max_depth=5, min_child_weight=1,
                                                 gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',
                                                 nthread=4, scale_pos_weight=1, seed=27)
                    model.fit(trnX, trnY)
                    prdY = model.predict(tstX)

                    # EVALUATION
                    fold_accuracies.append(metrics.accuracy_score(tstY, prdY))
                    fold = fold +1

                if statistics.mean(fold_accuracies) > macro_accuracy:
                    macro_accuracy = statistics.mean(fold_accuracies)
                    best_solution['learning_rate'] = lr
                    best_solution['n_estimators'] = n
        return best_solution['learning_rate'], best_solution['n_estimators']

    def depth_min_child_weight_tuning(self, data, labels, num_features):
        best_solution = {}
        macro_accuracy = 0
        skf = StratifiedKFold(n_splits=self.param_grid.get('fold'))
        for d in self.param_grid.get('max_depth'):
            for w in self.param_grid.get('min_child_weight'):
                fold_accuracies = []
                fold = 0
                single_indices = [i for i in range(0, len(labels), 3)]
                single_data = data.iloc[single_indices]
                single_labels = labels.iloc[single_indices]
                for train_index, test_index in skf.split(single_data, single_labels):
                    trnX, tstX, trnY, tstY = self.separate_and_prepare_data(data, labels, train_index, test_index, num_features)

                    # CLASSIFICATION
                    model = XGBClassifier(learning_rate =best_solution.get('learning_rate'), n_estimators= best_solution.get('n_estimators'),
                                          max_depth=d, min_child_weight=w, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                          objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
                    model.fit(trnX, trnY)
                    prdY = model.predict(tstX)

                    # EVALUATION
                    fold_accuracies.append(metrics.accuracy_score(tstY, prdY))
                    fold = fold +1
                if statistics.mean(fold_accuracies) > macro_accuracy:
                    macro_accuracy = statistics.mean(fold_accuracies)
                    best_solution['max_depth'] = d
                    best_solution['min_child_weight'] = w
        return best_solution['max_depth'], best_solution['min_child_weight']

    def gamma_tuning(self, data, labels, num_features):
        macro_accuracy = 0
        values_acc = {}
        values_sens = {}
        gamma = 0
        skf = StratifiedKFold(n_splits=self.param_grid.get('fold'))
        fig_acc, axs_acc = plt.subplots(1, 2, figsize=(13, 4), squeeze=False)
        fig_sens, axs_sens = plt.subplots(1, 2, figsize=(13, 4), squeeze=False)
        for g in self.param_grid.get('gamma'):
            fold_accuracies = []
            fold_sensitivities = []
            fold = 0
            single_indices = [i for i in range(0, len(labels), 3)]
            single_data = data.iloc[single_indices]
            single_labels = labels.iloc[single_indices]
            for train_index, test_index in skf.split(single_data, single_labels):
                trnX, tstX, trnY, tstY = self.separate_and_prepare_data(data, labels, train_index, test_index, num_features)

                # CLASSIFICATION
                model = XGBClassifier(learning_rate=self.best_solution.get('learning_rate'), n_estimators=self.best_solution.get('n_estimators'),
                                      max_depth=self.best_solution.get('max_depth'), min_child_weight=self.best_solution.get('min_child_weight'),
                                      gamma=g, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                                      nthread=4, scale_pos_weight=1, seed=27)
                model.fit(trnX, trnY)
                prdY = model.predict(tstX)

                # EVALUATION
                fold_accuracies.append(metrics.accuracy_score(tstY, prdY))
                fold_sensitivities.append(metrics.recall_score(tstY, prdY))
                fold = fold + 1
            values_acc[g] = statistics.mean(fold_accuracies)
            values_sens[g] = statistics.mean(fold_sensitivities)
            if statistics.mean(fold_accuracies) > macro_accuracy:
                macro_accuracy = statistics.mean(fold_accuracies)
                gamma = g
        plot.line_chart(axs_acc[0, 0], pd.Series(values_acc),
                                 'XGBoost depending on gamma',
                                 'gamma',
                                 'accuracy',
                                 percentage=True)
        plt.show()
        return gamma
