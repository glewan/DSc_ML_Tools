import statistics

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import sklearn.metrics as metrics
import arff #from liac-arff package
import pandas as pd, numpy as np
from IPython.display import display, HTML
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer #for dummification
from mlxtend.frequent_patterns import apriori, association_rules #for ARM
from sklearn import preprocessing, cluster
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.feature_selection import f_classif, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr, VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.decomposition import PCA
import warnings
from subprocess import call

from xgboost import XGBClassifier

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)
    from imblearn.over_sampling import SMOTE


def multiple_bar_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str, percentage=False):

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x = np.arange(len(xvalues))  # the label locations
    ax.set_xticks(x)
    ax.set_xticklabels(xvalues, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    width = 0.8  # the width of the bars
    step = width / len(yvalues)
    k = 0
    for name, y in yvalues.items():
        ax.bar(x + k * step, y, step, label=name, align='edge')
        k += 1
    ax.legend(loc='lower center', ncol=len(yvalues), bbox_to_anchor=(0.5, -0.2), fancybox = True, shadow = True)

    return

def display_balance_result(values):
    plt.figure()
    multiple_bar_chart(plt.gca(),
                       ['Sick', 'Healthy'],
                       values, 'Target', 'frequency', 'Class balance')
    plt.show()
    return

def variables_range(set):
    # #check variables range
    num_columns = set.shape[1]

    ranges = np.zeros((num_columns,2))
    i = 0
    for column in set.columns:
        ranges[i,0] = np.min(set[column].values)
        ranges[i,1] = np.max(set[column].values)
        #print("min:", ranges[i,0],  "\t\tmax", ranges[i,1])
        i += 1

    return ranges

# function that plots normalization efect
def describe_norm_efect(before_norm, after_norm):
    indices = np.arange(start=0, stop=before_norm.shape[0])
    array = np.concatenate((before_norm, after_norm), axis=1)
    min_max_dataframa = pd.DataFrame(np.sort(array, axis=0))
    min_max_dataframa.columns = ['min_values', 'max_values', 'min_val_norm', 'max_val_norm']
    print(min_max_dataframa.describe())

    return

def normalization(x_train, x_test):

    before_norm = variables_range(x_train)
    indices = x_train.index
    columns = x_train.columns

    # Normalize training data
    std_scale = preprocessing.StandardScaler().fit(x_train)
    # get list of ranges after normaization
    x_train_norm = std_scale.transform(x_train)


    # Transform normalized data to data frame (from ndarray)
    x_train = pd.DataFrame(x_train_norm, index=indices, columns=columns)


    after_norm = variables_range(x_train)

    indices = x_test.index
    columns = x_test.columns
    # Normalize testing data
    x_test_norm = std_scale.transform(x_test)
    x_test = pd.DataFrame(x_test_norm, index=indices, columns=columns)

    # describe_norm_efect(before_norm, after_norm)

    return x_train, x_test

def correlation_removal(train_x):

    # Using Pearson Correlation
    cor = train_x.corr()

    columns_names = train_x.columns
    uncorrelated_variables = columns_names
    for column in columns_names:
        # correlation with output variable
        cor_target = abs(cor[column])

        # selecting highly correlated features
        correlated_features = cor_target[cor_target >= 0.9]
        for index, correlation in correlated_features.iteritems():

            if index != column and index in uncorrelated_variables:
                uncorrelated_variables = uncorrelated_variables.drop(index)

    #print("Number of removed perfectly correlated variables", len(columns_names) - len(uncorrelated_variables))

    return train_x[uncorrelated_variables]

def display_histogram(x_train, y_train, attribute_name):
    # auxiliary dataframe to connect observations with class

    df_aux = pd.concat([x_train, y_train], axis=1)

    plot = sns.FacetGrid(df_aux, hue='class', height = 5, legend_out=True) \
            .map(sns.distplot,attribute_name) \
            .add_legend();
    new_title = 'Class'
    plot._legend.set_title(new_title)
    new_labels = ['Healty', 'Sick']
    for t, l in zip(plot._legend.texts, new_labels): t.set_text(l)

    plt.show()

    return

def generate_features_ranking(x_train, y_train, type):
    if type == '1':
        # apply SelectKBest class to extract top 10 best features
        selector = SelectKBest(score_func=f_classif, k=10)
        X_new = selector.fit_transform(x_train, y_train)
        df_scores = pd.DataFrame(selector.scores_)
        df_columns = pd.DataFrame(x_train.columns)

        #print("Scores:", selector.scores_)
        # concat two dataframes for better visualization
        attributesScores = pd.concat([df_columns, df_scores], axis=1)
        # name new columns
        attributesScores.columns = ['Attr', 'Score']
        #print("Original data space:\n", x[0:3], "\nNew data space:\n", X_new[0:3])
        #print("ANOVA test based ranking")
        #print(attributesScores.nlargest(10, 'Score'))

        return attributesScores.sort_values(by='Score', axis=0, ascending=False)

    elif type == '2':

        selector = SelectPercentile(mutual_info_regression, percentile=30)
        X_new = selector.fit_transform(x_train, y_train)
        df_scores = pd.DataFrame(selector.scores_)
        df_columns = pd.DataFrame(x_train.columns)
        attributesScores = pd.concat([df_columns, df_scores], axis=1)
        # name new columns
        attributesScores.columns = ['Attr', 'Score']
        #print("Original data space:\n", x[0:3], "\nNew data space:\n", X_new[0:3])
        #print("ANOVA test based ranking")
        #print(attributesScores.nlargest(10, 'Score'))

        return attributesScores.sort_values(by='Score', axis=0, ascending=False)
        # ranking = attributesScores_percentile.sort_values(by='Pvalues', axis=0)
    else:
        ValueError("wrong type")


def bayesian_naive_classify(trnX, tstX, trnY, tstY, labels):
    # print(data.shape)
    clf = GaussianNB()
    clf.fit(trnX, trnY)
    prdY = clf.predict(tstX)
    cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)
    #func.plot_confusion_matrix2(cnf_mtx, labels)

    # func.plot_confusion_matrix(plt.gca(), cnf_mtx, labels)
    return cnf_mtx

def display_best_feature(x_train, y_train, ranking, n):
    i = 0
    for attribute in ranking['Attr']:
        display_histogram(x_train, y_train, attribute)
        i += 1
        if i == n:
            break
    return

def bayes_n_features(x_train, x_test, y_train, y_test, labels, ranking, numbers_of_features):

    accuracy_rates = []
    for number in numbers_of_features:
        attributes = ranking.head(number)
        # print("x_train", x_train.shape)
        # print("x_test", x_test.shape)
        # print("y_train", y_train.shape)
        # print("y_test", y_test.shape)

        cnf_mtx = bayesian_naive_classify(x_train[attributes['Attr']], x_test[attributes['Attr']], y_train, y_test, labels)
        accuracy = (cnf_mtx[0][0] + cnf_mtx[1][1]) / np.sum(cnf_mtx)
        accuracy_rates.append(accuracy)

    #print("max accuracy:", max(accuracy_rates), "for number of attributes:", numbers_of_features[accuracy_rates.index(max(accuracy_rates))])

    return accuracy_rates

def print_balance(target_count_before, target_count_after, title1, title2):
    # plot balance bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title(title1)
    ax2.set_title(title2)
    ax1.bar(target_count_before.index, target_count_before.values)
    ax1.set_ylabel('Observations quantity')
    ax2.bar(target_count_after.index, target_count_after.values)

    plt.show()

    min_class = target_count_before.idxmin()
    ind_min_class = target_count_before.index.get_loc(min_class)


    # # print data from before performing balance
    # print('Minority class:', target_count_before[ind_min_class])
    # print('Majority class:', target_count_before[1 - ind_min_class])
    # print('Proportion:', round(target_count_before[ind_min_class] / target_count_before[1 - ind_min_class], 2), ': 1')
    #
    # min_class = target_count_after.idxmin()
    # ind_min_class = target_count_after.index.get_loc(min_class)
    #
    # # print data from after performing balance
    # print('Minority class:', target_count_after[ind_min_class])
    # print('Majority class:', target_count_after[1 - ind_min_class])
    # print('Proportion:', round(target_count_after[ind_min_class] / target_count_after[1 - ind_min_class], 2), ': 1')

    return


def split_back(dataset):
    y_train = dataset.pop('class')
    x_train = dataset

    return x_train, y_train


def perform_balance_undersampling(train_x, train_y, plot_data=False):

    data = pd.concat([train_x, train_y], axis = 1)
    #  get class balance before performing balance
    target_count_before = data.replace(to_replace={'class':{0:'Healthy', 1:'Sick'}})['class'].value_counts()

    #undersample majority class - keep only one observation per sick patient
    sick = data[data['class'] == 1]
    sick = sick.iloc[::3, :]
    healthy = data[data['class'] == 0]
    # get equivalent number of healthy 'observations'
    healthy = healthy.iloc[:sick.shape[0], :]

    data_new = pd.concat([sick, healthy], axis=0)

    target_count_after = data_new.replace(to_replace={'class': {0: 'Healthy', 1: 'Sick'}})['class'].value_counts()
    if plot_data:
        #print_balance(target_count_before, target_count_after, title1='Before undersampling', title2='Class balance after')
        pass



    return split_back(data_new)


# function that ruturns n- best(based on ranking) features as dataset
def return_substets(x_train, ranking, number):

    attributes = ranking.head(number)

    return x_train[attributes['Attr']]

def plot_rankings(numbers_of_features, ranking_accuracy_rates):
    #print(accuracy_rates)
    plt.figure(figsize=(10, 6))
    for ranking in ranking_accuracy_rates:
        plt.plot(numbers_of_features, ranking)

    plt.ylabel('accuracy rate ANOVA ranking ')
    plt.xlabel('num of features')
    plt.legend(labels=['oversampled', 'undersampled', 'SMOTE'])
    plt.grid()
    plt.show()

    return


def balance_oversampling(x_train, y_train):
    # concatenate back data
    X = pd.concat([x_train, y_train], axis=1)

    # seperate minority and majority class
    sick = X[X['class']==1]
    healthy =  X[X['class']==0]

    # upsample minority
    healthy_upsampled = resample(healthy,
                               replace=True,  # sample with replacement
                               n_samples=len(sick),  # match number in majority class
                               random_state=27)  # reproducible results

    # combine majority and upsampled minority
    upsampled = pd.concat([sick, healthy_upsampled])

    return split_back(upsampled)

def balance_with_SMOTE(x_train, y_train):
    data = pd.concat([x_train, y_train], axis=1)
    target_count = data['class'].value_counts()
    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)
    smote = SMOTE(ratio='minority')


    smote_x, smote_y = smote.fit_sample(x_train, y_train)
    X = pd.DataFrame(smote_x, columns=x_train.columns)
    X['class'] = smote_y

    return split_back(X)

def discretize_width(set, num, columns_to_skip):
    # discretize real-valued attributes: width

    newdf = set.copy()
    for col in newdf:

        if col not in columns_to_skip:
            tmp = pd.cut(newdf[col], num, precision=0)
            num_labels = len(pd.unique(tmp))
            newdf[col] = pd.cut(newdf[col], num, precision=0, labels=[str(x) for x in list(range(num))])

    return newdf

def discretize_depth(set, num, columns_to_skip):

    labels = []
    for i in range(1,num):
        labels.append(str(i))

    # discretize real-valued attributes: depth
    newdf = set.copy()
    for col in newdf:
        if col not in columns_to_skip:
            newdf[col] = pd.qcut(newdf[col], num, precision=0, duplicates='drop', labels=labels)

    return newdf

# return list of selected columns per groups
def unsupervised_feature_selection(data, treshhold):
    # unsupervised feature selection
    groups_parkinson = [0, 56, 140, 322, 755]
    columns_within_groups = []
    for i in range(len(groups_parkinson) - 1):
        columns_to_analyze = data.columns[groups_parkinson[i]:groups_parkinson[i + 1]]
        X = data[columns_to_analyze]
        # print("Number of features =", X.shape[1])
        selector = VarianceThreshold()
        selector.fit(X)
        values = selector.variances_
        zipped =  zip(values, X.columns)
        sorted_zip = sorted(zipped, key=lambda tup: tup[0])


        # print("Feature variance =", values)
        columns_to_append = []
        for x in sorted_zip:
            if x[0] >= treshhold:
                columns_to_append.append(x[1])

        # selector = VarianceThreshold(threshold=treshhold)
        # X_new = selector.fit_transform(X)

        # print("Number of features =", X_new.shape[1])
        columns_within_groups.append(columns_to_append)

    num_columns_removed = data.shape[1]
    for x in columns_within_groups:
        num_columns_removed -= len(x)

    print("Number of columns removed in unsupervised approach", num_columns_removed)

    return columns_within_groups

#--------------------------------------------------------------------------

def correlation_removal_kcross(train_x, test_x):
    # Using Pearson Correlation
    cor = train_x.corr()

    columns_names = train_x.columns
    uncorrelated_variables = columns_names
    for column in columns_names:
        # correlation with output variable
        cor_target = abs(cor[column])

        # selecting highly correlated features
        correlated_features = cor_target[cor_target >= 0.9]
        for index, correlation in correlated_features.iteritems():

            if index != column and index in uncorrelated_variables:
                uncorrelated_variables = uncorrelated_variables.drop(index)

    return train_x[uncorrelated_variables], test_x[uncorrelated_variables]

def show_tree(trnX, trnY):
    from sklearn.tree import export_graphviz

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(trnX, trnY)

    dot_data = export_graphviz(tree, out_file='dtree.dot', filled=True, rounded=True, special_characters=True)
    # Convert to png
    from subprocess import call
    call(['dot', '-Tpng', '../dtree.dot', '-o', './dtree.png', '-Gdpi=600'])

    plt.figure(figsize=(14, 18))
    plt.imshow(plt.imread('./dtree.png'))
    plt.axis('off')
    plt.show()

def line_chart(ax: plt.Axes, series: pd.Series, title: str, xlabel: str, ylabel: str, percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.plot(series)
def plot_confusion_matrix(ax: plt.Axes, cnf_matrix: np.ndarray, legend_title: str, classes_names: list, normalize: bool = False):
    if normalize:
        total = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / total
        title = legend_title
    else:
        cm = cnf_matrix
        title = legend_title
    np.set_printoptions(precision=2)
    tick_marks = np.arange(0, len(classes_names), 1)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    fmt = '.2f' # if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center")

def multiple_line_chart_cross(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, legend_title: str, xlabel: str, ylabel: str, percentage=False):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)

    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    # ax.legend(legend, title=legend_title, loc='center left', bbox_to_anchor=(1, 0.8))
    ax.legend(legend, title=legend_title, loc='best')

def separate_and_prepare_data(data, labels, train_index, test_index, num_features, method):
    trn_X, tst_X = data[data.index.isin(train_index)], data[data.index.isin(test_index)]
    trn_Y, tst_Y = labels[labels.index.isin(train_index)], labels[labels.index.isin(test_index)]

    # PREPARATION FOR TRAINING PART
    trn_X, tst_X = normalization(trn_X, tst_X)
    if method == 1:
        trn_X, trn_Y = balance_oversampling(trn_X, trn_Y)
    elif method == 2:
        trn_X, trn_Y = perform_balance_undersampling(trn_X, trn_Y)
    else:
        trn_X, trn_Y = balance_with_SMOTE(trn_X, trn_Y)
    trn_X, tst_X = correlation_removal_kcross(trn_X, tst_X)
    trn_X, tst_X = select_features(trn_X, trn_Y, tst_X, num_features)

    trnX = trn_X.values
    tstX = tst_X.values
    trnY = trn_Y.values
    tstY = tst_Y.values

    return trnX, tstX, trnY, tstY

def select_features(x_train, y_train, x_test, n_features, ranking_type='1' ):
    ranking = generate_features_ranking(x_train, y_train, ranking_type)
    attributes = ranking.head(n_features)

    return x_train[attributes['Attr']], x_test[attributes['Attr']]

def compute_confm_values(fold_predictions):
    TN = sum(fold_predictions[:, 0])
    FP = sum(fold_predictions[:, 1])
    FN = sum(fold_predictions[:, 2])
    TP = sum(fold_predictions[:, 3])

    return TN, FP, FN, TP

def naive_Bayes(data, labels, num_features, method):
    param_grid = {'estimators': {'GaussianNB': GaussianNB()},
                  'fold': 5}

    best_solution = {'estimator': 'noname',
                     'accuracy': 0,
                     'sensitivity': 0,
                     'confusion_matrix': [0, 0, 0, 0]}

    skf = StratifiedKFold(n_splits=param_grid.get('fold'))

    for clf in param_grid.get('estimators'):
        fold_accuracies = []
        fold_sensitivities = []
        fold_predictions = np.empty((param_grid.get('fold'), 4))
        fold = 0
        single_indices = [i for i in range(0, len(labels), 3)]
        single_data = data.iloc[single_indices]
        single_labels = labels.iloc[single_indices]
        for train_index, test_index in skf.split(single_data, single_labels):
            trnX, tstX, trnY, tstY = separate_and_prepare_data(data, labels, train_index, test_index, num_features, method)

            # CLASSIFICATION
            param_grid.get('estimators')[clf].fit(trnX, trnY)
            prdY = param_grid.get('estimators')[clf].predict(tstX)

            # EVALUATION
            fold_accuracies.append(metrics.accuracy_score(tstY, prdY))
            fold_sensitivities.append(metrics.recall_score(tstY, prdY))
            tn, fp, fn, tp = confusion_matrix(tstY, prdY).ravel()
            fold_predictions[fold] = [tn, fp, fn, tp]
            fold = fold + 1

        if (statistics.mean(fold_accuracies) > best_solution.get('accuracy')):
            best_solution['estimator'] = clf
            best_solution['accuracy'] = statistics.mean(fold_accuracies)
            best_solution['sensitivity'] = statistics.mean(fold_sensitivities)

            TN, FP, FN, TP = compute_confm_values(fold_predictions)
            best_solution['confusion_matrix'] = np.array(([TN, FP], [FN, TP]))

    # fig, axs = plt.subplots(1, 1, figsize=(4, 4), squeeze=False)
    # plot_confusion_matrix(axs[0, 0], best_solution.get('confusion_matrix'), 'Confusion matrix', [0, 1], True)
    # plt.show()
    return best_solution

def show_tree(trnX, trnY, best_solution):
    from sklearn.tree import export_graphviz

    tree = DecisionTreeClassifier(min_samples_leaf=best_solution.get('min_samples_leaf'),
                                  max_depth=best_solution.get('max_depth'),
                                  criterion=best_solution.get('criteria'),
                                  min_impurity_decrease=0.005)
    tree.fit(trnX, trnY)

    dot_data = export_graphviz(tree, out_file='dtree.dot', filled=True, rounded=True, special_characters=True)
    # Convert to png
    from subprocess import call
    call(['dot', '-Tpng', 'dtree.dot', '-o', 'dtree.png', '-Gdpi=600'])
    plt.figure(figsize=(14, 18))
    plt.imshow(plt.imread('dtree.png'))
    plt.axis('off')
    plt.show()

def investigate_decision_tree(data, labels, num_features, method):
    param_grid = {'min_samples_leaf': [.05, .025, .01, .0075, .005, .0025, .001],
                  'max_depths': [5, 10, 25, 50],
                  'criteria': ['entropy', 'gini'],
                  'fold': 5}

    best_solution = {'criteria': 'noname',
                     'min_samples_leaf': 0,
                     'max_depths': 0,
                     'accuracy': 0,
                     'sensitivity': 0,
                     'confusion_matrix': [0, 0, 0, 0]}

    fig_acc, axs_acc = plt.subplots(1, 2, figsize=(13, 4), squeeze=False)
    fig_sens, axs_sens = plt.subplots(1, 2, figsize=(13, 4), squeeze=False)

    skf = StratifiedKFold(n_splits=param_grid.get('fold'))

    for k in range(len(param_grid.get('criteria'))):
        f = param_grid.get('criteria')[k]
        values_acc = {}
        values_sens = {}
        for d in param_grid.get('max_depths'):
            accuracies_values = []
            sensitivities_values = []
            for n in param_grid.get('min_samples_leaf'):
                tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f,
                                              min_impurity_decrease=0.005)
                fold_accuracies = []
                fold_sensitivities = []
                fold_predictions = np.empty((param_grid.get('fold'), 4))
                fold = 0
                single_indices = [i for i in range(0, len(labels), 3)]
                single_data = data.iloc[single_indices]
                single_labels = labels.iloc[single_indices]
                for train_index, test_index in skf.split(single_data, single_labels):
                    trnX, tstX, trnY, tstY = separate_and_prepare_data(data, labels, train_index, test_index,
                                                                       num_features, method)

                    # CLASSIFICATION
                    tree.fit(trnX, trnY)
                    prdY = tree.predict(tstX)

                    # EVALUATION
                    fold_accuracies.append(metrics.accuracy_score(tstY, prdY))
                    fold_sensitivities.append(metrics.recall_score(tstY, prdY))
                    tn, fp, fn, tp = confusion_matrix(tstY, prdY).ravel()
                    fold_predictions[fold] = [tn, fp, fn, tp]
                    fold = fold + 1
                if (statistics.mean(fold_accuracies) > best_solution.get('accuracy')):
                    best_solution['criteria'] = param_grid.get('criteria')[k]
                    best_solution['min_samples_leaf'] = n
                    best_solution['max_depths'] = d
                    best_solution['accuracy'] = statistics.mean(fold_accuracies)
                    best_solution['sensitivity'] = statistics.mean(fold_sensitivities)

                    TN, FP, FN, TP = compute_confm_values(fold_predictions)
                    best_solution['confusion_matrix'] = np.array(([TN, FP], [FN, TP]))
                accuracies_values.append(statistics.mean(fold_accuracies))
                sensitivities_values.append(statistics.mean(fold_sensitivities))

            values_acc[d] = accuracies_values
            values_sens[d] = sensitivities_values
        multiple_line_chart_cross(axs_acc[0, k], param_grid.get('min_samples_leaf'), values_acc,
                                 'Decision Trees with %s criteria' % f,
                                 'max_depths',
                                 'min_samples_leaf',
                                 'accuracy',
                                 percentage=True)

        multiple_line_chart_cross(axs_sens[0, k], param_grid.get('min_samples_leaf'), values_sens,
                                 'Decision Trees with %s criteria' % f,
                                 'max_depths',
                                 'min_samples_leaf',
                                 'sensitivity',
                                 percentage=True)
    plt.show()
    fig, axs = plt.subplots(1, 1, figsize=(4, 4), squeeze=False)
    plot_confusion_matrix(axs[0, 0], best_solution.get('confusion_matrix'), 'Confusion matrix', [0, 1], True)
    plt.show()
    return best_solution

def investigate_KNN(data, labels, num_features, method):
    param_grid = {'dist': ['manhattan', 'euclidean', 'chebyshev'],
                  'n_neighbors': np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 25, 30, 50, 70]),
                  'fold': 5}

    best_solution = {'dist': 'noname',
                     'n_neighbors': 0,
                     'accuracy': 0,
                     'sensitivity': 0,
                     'confusion_matrix': [0, 0, 0, 0]}

    fig_acc, axs_acc = plt.subplots(1, 1, figsize=(6, 4), squeeze=False)
    fig_sens, axs_sens = plt.subplots(1, 1, figsize=(6, 4), squeeze=False)

    skf = StratifiedKFold(n_splits=param_grid.get('fold'))

    values_acc = {}
    values_sens = {}
    for i, d in enumerate(param_grid.get('dist')):
        accuracies_values = []
        sensitivities_values = []
        for n in param_grid.get('n_neighbors'):
            fold_accuracies = []
            fold_sensitivities = []
            fold_predictions = np.empty((param_grid.get('fold'), 4))
            fold = 0
            single_indices = [i for i in range(0, len(labels), 3)]
            single_data = data.iloc[single_indices]
            single_labels = labels.iloc[single_indices]
            for train_index, test_index in skf.split(single_data, single_labels):
                trnX, tstX, trnY, tstY = separate_and_prepare_data(data, labels, train_index, test_index, num_features, method)

                # CLASSIFICATION
                knn = KNeighborsClassifier(n_neighbors=n, metric=d)
                knn.fit(trnX, trnY)
                prdY = knn.predict(tstX)

                # EVALUATION
                fold_accuracies.append(metrics.accuracy_score(tstY, prdY))
                fold_sensitivities.append(metrics.recall_score(tstY, prdY))
                tn, fp, fn, tp = confusion_matrix(tstY, prdY).ravel()
                fold_predictions[fold] = [tn, fp, fn, tp]
                fold = fold + 1
            if (statistics.mean(fold_accuracies) > best_solution.get('accuracy')):
                best_solution['dist'] = param_grid.get('dist')[i]
                best_solution['n_neighbors'] = n
                best_solution['accuracy'] = statistics.mean(fold_accuracies)
                best_solution['sensitivity'] = statistics.mean(fold_sensitivities)

                TN, FP, FN, TP = compute_confm_values(fold_predictions)
                best_solution['confusion_matrix'] = np.array(([TN, FP], [FN, TP]))
            # result for different number of neighbours
            accuracies_values.append(statistics.mean(fold_accuracies))
            sensitivities_values.append(statistics.mean(fold_sensitivities))
        # results for every distance with different num of neighbours
        values_acc[d] = accuracies_values
        values_sens[d] = sensitivities_values
    multiple_line_chart_cross(axs_acc[0, 0], param_grid.get('n_neighbors'), values_acc,
                             'KNN for different number of neighbours',
                             'Distance metrics',
                             'nr neighbours',
                             'accuracy',
                             percentage=True)
    multiple_line_chart_cross(axs_sens[0, 0], param_grid.get('n_neighbors'), values_sens,
                             'KNN for different number of neighbours',
                             'Distance metrics',
                             'nr neighbours',
                             'sensitivity',
                             percentage=True)
    plt.show()
    fig, axs = plt.subplots(1, 1, figsize=(4, 4), squeeze=False)
    plot_confusion_matrix(axs[0, 0], best_solution.get('confusion_matrix'), 'Confusion matrix', [0, 1], True)
    plt.show()
    return best_solution

def investigate_random_forest(data, labels, num_features, method):
    param_grid = {'n_estimators': [5, 7, 10, 17, 25, 30, 50, 75, 100, 150, 200, 300],
                  'max_depths': [5, 10, 15, 25, 50],
                  'max_features': ['sqrt', 'log2'],
                  'fold': 5}

    best_solution = {'n_estimators': 0,
                     'max_depths': 0,
                     'max_features': 'noname',
                     'accuracy': 0,
                     'sensitivity': 0,
                     'confusion_matrix': [0, 0, 0, 0]}

    fig_acc, axs_acc = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    fig_sens, axs_sens = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)

    skf = StratifiedKFold(n_splits=param_grid.get('fold'))

    for k in range(len(param_grid.get('max_features'))):
        f = param_grid.get('max_features')[k]
        values_acc = {}
        values_sens = {}
        for d in param_grid.get('max_depths'):
            accuracies_values = []
            sensitivities_values = []
            for n in param_grid.get('n_estimators'):
                fold_accuracies = []
                fold_sensitivities = []
                fold_predictions = np.empty((param_grid.get('fold'), 4))
                fold = 0
                single_indices = [i for i in range(0, len(labels), 3)]
                single_data = data.iloc[single_indices]
                single_labels = labels.iloc[single_indices]
                for train_index, test_index in skf.split(single_data, single_labels):
                    trnX, tstX, trnY, tstY = separate_and_prepare_data(data, labels, train_index, test_index,
                                                                       num_features, method)

                    # CLASSIFICATION
                    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                    rf.fit(trnX, trnY)
                    prdY = rf.predict(tstX)

                    # EVALUATION
                    fold_accuracies.append(metrics.accuracy_score(tstY, prdY))
                    fold_sensitivities.append(metrics.recall_score(tstY, prdY))
                    tn, fp, fn, tp = confusion_matrix(tstY, prdY).ravel()
                    fold_predictions[fold] = [tn, fp, fn, tp]
                    fold = fold + 1

                if (statistics.mean(fold_accuracies) > best_solution.get('accuracy')):
                    best_solution['n_estimators'] = n
                    best_solution['max_depths'] = d
                    best_solution['max_features'] = f
                    best_solution['accuracy'] = statistics.mean(fold_accuracies)
                    best_solution['sensitivity'] = statistics.mean(fold_sensitivities)

                    TN, FP, FN, TP = compute_confm_values(fold_predictions)
                    best_solution['confusion_matrix'] = np.array(([TN, FP], [FN, TP]))

                accuracies_values.append(statistics.mean(fold_accuracies))
                sensitivities_values.append(statistics.mean(fold_sensitivities))

            values_acc[d] = accuracies_values
            values_sens[d] = sensitivities_values
        multiple_line_chart_cross(axs_acc[0, k], param_grid.get('n_estimators'), values_acc,
                                 'Random Forests with %s features' % f,
                                 'max_depths',
                                 'nr estimators',
                                 'accuracy',
                                 percentage=True)
        plt.figure()
        multiple_line_chart_cross(axs_sens[0, k], param_grid.get('n_estimators'), values_sens,
                                 'Random Forests with %s features' % f,
                                 'max_depths',
                                 'nr estimators',
                                 'sensitivity',
                                 percentage=True)
    plt.show()
    fig, axs = plt.subplots(1, 1, figsize=(4, 4), squeeze=False)
    plot_confusion_matrix(axs[0, 0], best_solution.get('confusion_matrix'), 'Confusion matrix', [0, 1], True)
    plt.show()
    return best_solution

def investigate_num_features_for_knn(data, labels, method):
    num_features = range(2,30, 2)
    neighbours = [2, 5, 10, 20]
    fig, axs = plt.subplots(1, 1, figsize=(6, 4), squeeze=False)
    acc_for_neighbours = {}
    for n in neighbours:
        accuracies_values = []
        for f in num_features:
            skf = StratifiedKFold(n_splits=5)
            fold_accuracies = []
            fold_sensitivities = []
            single_indices = [i for i in range(0, len(labels), 3)]
            single_data = data.iloc[single_indices]
            single_labels = labels.iloc[single_indices]
            for train_index, test_index in skf.split(single_data, single_labels):
                trnX, tstX, trnY, tstY = separate_and_prepare_data(data, labels, train_index, test_index,
                                                                   f, method)
                knn = KNeighborsClassifier(n_neighbors=n, metric='chebyshev')
                # CLASSIFICATION
                knn.fit(trnX, trnY)
                prdY = knn.predict(tstX)

                # EVALUATION
                fold_accuracies.append(metrics.accuracy_score(tstY, prdY))
                fold_sensitivities.append(metrics.recall_score(tstY, prdY))

            accuracies_values.append(statistics.mean(fold_accuracies))
        acc_for_neighbours[n] = accuracies_values
    multiple_line_chart_cross(axs[0, 0], num_features, acc_for_neighbours,
                        'KNN for different number of features',
                        'Number of neighbours',
                        'num features',
                        'accuracy',
                        percentage=True)
    plt.grid()
    plt.show()

param_grid = {
    "learning_rate": [0.01, 0.3, 0.5],
    "n_estimators": [5, 7, 10, 17, 25, 30, 50],
    "max_depth": [3, 5, 7, 10],
    "min_child_weight": [1, 2, 5],
    'gamma': [0, 0.5, 2, 4, 7, 10],
    "subsample": [0.7, 0.8, 1],
    "colsample_bytree": [0.7],
    "objective": 'binary:logistic',
    "fold": 5}

best_solution = {}
def select_params(data, labels):
    lr_num_estimators_tuning(data, labels, 20)
    depth_min_child_weight_tuning(data, labels, 20)
    gamma_tuning(data, labels, 20)

def investigate_xg_boost(data, labels, num_features):
    select_params(data, labels)
    skf = StratifiedKFold(n_splits=param_grid.get('fold'))
    fold_accuracies = []
    fold_sensitivities = []
    fold_predictions = np.empty((param_grid.get('fold'), 4))
    fold = 0
    single_indices = [i for i in range(0, len(labels), 3)]
    single_data = data.iloc[single_indices]
    single_labels = labels.iloc[single_indices]
    for train_index, test_index in skf.split(single_data, single_labels):
        trnX, tstX, trnY, tstY = separate_and_prepare_data(data, labels, train_index, test_index, num_features, 1)

        # CLASSIFICATION
        model = XGBClassifier(learning_rate=best_solution.get('learning_rate'),
                              n_estimators=best_solution.get('n_estimators'),
                              max_depth=best_solution.get('max_depth'),
                              min_child_weight=best_solution.get('min_child_weight'),
                              gamma=best_solution.get('gamma'),
                              subsample=0.8,
                              colsample_bytree=0.8,
                              objective='binary:logistic',
                              nthread=4,
                              scale_pos_weight=1,
                              seed=27)
        model.fit(trnX, trnY)
        prdY = model.predict(tstX)

        # EVALUATION
        fold_accuracies.append(metrics.accuracy_score(tstY, prdY))
        fold_sensitivities.append(metrics.recall_score(tstY, prdY))
        tn, fp, fn, tp = confusion_matrix(tstY, prdY).ravel()
        fold_predictions[fold] = [tn, fp, fn, tp]
        fold = fold + 1
    TN, FP, FN, TP = compute_confm_values(fold_predictions)

    best_solution['confusion_matrix'] = np.array(([TN, FP], [FN, TP]))
    best_solution['accuracy'] = statistics.mean(fold_accuracies)
    best_solution['sensitivity'] = statistics.mean(fold_sensitivities)
    return best_solution

def lr_num_estimators_tuning(data, labels, num_features):
    macro_accuracy = 0
    skf = StratifiedKFold(n_splits=param_grid.get('fold'))
    for lr in param_grid.get('learning_rate'):
        for n in param_grid.get('n_estimators'):
            fold_accuracies = []
            fold = 0
            single_indices = [i for i in range(0, len(labels), 3)]
            single_data = data.iloc[single_indices]
            single_labels = labels.iloc[single_indices]
            for train_index, test_index in skf.split(single_data, single_labels):
                trnX, tstX, trnY, tstY = separate_and_prepare_data(data, labels, train_index, test_index, num_features, 1)

                # CLASSIFICATION
                model = XGBClassifier(learning_rate =lr, n_estimators=n, max_depth=5, min_child_weight=1,
                                             gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',
                                             nthread=4, scale_pos_weight=1, seed=27)
                model.fit(trnX, trnY)
                prdY = model.predict(tstX)

                # EVALUATION
                fold_accuracies.append(metrics.accuracy_score(tstY, prdY))
                fold = fold +1

            if (statistics.mean(fold_accuracies) > macro_accuracy):
                macro_accuracy = statistics.mean(fold_accuracies)
                best_solution['learning_rate'] = lr
                best_solution['n_estimators'] = n

def depth_min_child_weight_tuning(data, labels, num_features):
    macro_accuracy = 0
    skf = StratifiedKFold(n_splits=param_grid.get('fold'))
    for d in param_grid.get('max_depth'):
        for w in param_grid.get('min_child_weight'):
            fold_accuracies = []
            fold = 0
            single_indices = [i for i in range(0, len(labels), 3)]
            single_data = data.iloc[single_indices]
            single_labels = labels.iloc[single_indices]
            for train_index, test_index in skf.split(single_data, single_labels):
                trnX, tstX, trnY, tstY = separate_and_prepare_data(data, labels, train_index, test_index, num_features, 1)

                # CLASSIFICATION
                model = XGBClassifier(learning_rate =best_solution.get('learning_rate'), n_estimators= best_solution.get('n_estimators'),
                                      max_depth=d, min_child_weight=w, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                      objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
                model.fit(trnX, trnY)
                prdY = model.predict(tstX)

                # EVALUATION
                fold_accuracies.append(metrics.accuracy_score(tstY, prdY))
                fold = fold +1
            if (statistics.mean(fold_accuracies) > macro_accuracy):
                macro_accuracy = statistics.mean(fold_accuracies)
                best_solution['max_depth'] = d
                best_solution['min_child_weight'] = w

def gamma_tuning(data, labels, num_features):
    macro_accuracy = 0
    values_acc = {}
    values_sens = {}
    skf = StratifiedKFold(n_splits=param_grid.get('fold'))
    fig, axs = plt.subplots(1, 1, figsize=(4, 4), squeeze=False)
    for g in param_grid.get('gamma'):
        fold_accuracies = []
        fold_sensitivities = []
        fold = 0
        single_indices = [i for i in range(0, len(labels), 3)]
        single_data = data.iloc[single_indices]
        single_labels = labels.iloc[single_indices]
        for train_index, test_index in skf.split(single_data, single_labels):
            trnX, tstX, trnY, tstY = separate_and_prepare_data(data, labels, train_index, test_index, num_features, 1)

            # CLASSIFICATION
            model = XGBClassifier(learning_rate=best_solution.get('learning_rate'), n_estimators=best_solution.get('n_estimators'),
                                  max_depth=best_solution.get('max_depth'), min_child_weight=best_solution.get('min_child_weight'),
                                  gamma=g, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                                  nthread=4, scale_pos_weight=1, seed=27)
            model.fit(trnX, trnY)
            prdY = model.predict(tstX)

            # EVALUATION
            fold_accuracies.append(metrics.accuracy_score(tstY, prdY))
            fold_sensitivities.append(metrics.recall_score(tstY, prdY))
            fold = fold +1
        values_acc[g] = statistics.mean(fold_accuracies)
        values_sens[g] = statistics.mean(fold_sensitivities)
        if (statistics.mean(fold_accuracies) > macro_accuracy):
            macro_accuracy = statistics.mean(fold_accuracies)
            best_solution['gamma'] = g
    line_chart(axs[0, 0], pd.Series(values_acc),
                             'XGBoost depending on gamma',
                             'gamma',
                             'accuracy',
                             percentage=True)
    # multiple_line_chart(axs[0, 0], param_grid.get('gamma'), values_sens,
    #                     'XGBoost in dependency on gamma',
    #                     'Metrics',
    #                     'gamma',
    #                     '',
    #                     percentage=True)
    plt.show()





if __name__ == '__main__':
    #################### READ DATA #############################

    # import data
    data_set_name = '../data/pd_speech_features.csv'
    data: pd.DataFrame = pd.read_csv(data_set_name, index_col='id', sep=',', header=1, decimal='.', parse_dates=True,
                   infer_datetime_format=True)

    # print(data.index)
    # print(data[data.index ==0])
    #################### SPLIT DATA #############################

    y = data.pop('class')
    x = data
    labels = pd.unique(y)
    indeces = pd.unique(x.index)

    # join back data(pop removes column)
    data = pd.concat([x,y], axis=1)

    # define auxiliary dataframe
    aux_df = pd.DataFrame(columns=data.columns)
    for index in indeces:
        aux_df = aux_df.append(data[data.index == index].iloc[0, :])

    yy = aux_df.pop('class')
    xx = aux_df

    x_train, x_test, y_train, y_test = train_test_split(xx, yy, train_size=0.7)

    train = pd.DataFrame(columns=data.columns)
    test = pd.DataFrame(columns=data.columns)
    for index in x_train.index:
        train = train.append((data[data.index == index]))

    for index in x_test.index:
        test = test.append((data[data.index == index]))

    y_train = train.pop('class')
    y_train = y_train.astype('int')
    x_train = train

    y_test = test.pop('class')
    y_test = y_test.astype('int')
    x_test = test


    #################### NORMALIZE DATA #############################

    target_count_dataset = y.replace({0: 'Healthy', 1: 'Sick'}).value_counts()
    target_count_test = y_test.replace({0: 'Healthy', 1: 'Sick'}).value_counts()

    #print_balance(target_count_dataset, target_count_test, title1='balance of whole set', title2='balance of test set')


    x_train, x_test = normalization(x_train, x_test)


    #################### DATA BALANCE #############################

    # befora sampling
    target_count = y_train.replace({0: 'Healthy', 1: 'Sick'}).value_counts()
    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)
    # auxiliary dictionary to plot data
    values = {'Original': [target_count['Healthy'], target_count['Sick']]}

    # SNODENT
    x_train_s, y_train_s = balance_with_SMOTE(x_train, y_train)

    # undersampling
    x_train_u, y_train_u = perform_balance_undersampling(x_train, y_train, plot_data=True)

    # oversamppling
    x_train_o, y_train_o = balance_oversampling(x_train, y_train)

    minor = y_train_s.replace({0: 'Healthy', 1: 'Sick'}).value_counts()

    values['SMOTE'] = [minor['Healthy'],minor['Sick']]

    minor = y_train.replace({0: 'Healthy', 1: 'Sick'}).value_counts()
    values['OVERSAMPLING'] = [minor['Healthy'], minor['Sick']]

    minor = y_train_u.replace({0: 'Healthy', 1: 'Sick'}).value_counts()
    values['UNDERSAMPLING'] = [minor['Healthy'], minor['Sick']]

    #print(values)
    #display_balance_result(values)
    #################### CORRELATED DATA #############################

    x_train_o_rem_sup = correlation_removal(x_train_o)
    x_train_u_rem_sup = correlation_removal(x_train_u)
    x_train_s_rem_sup = correlation_removal(x_train_s)

    #################### FEATURE SELECTION #############################

    ranking_kselect_anova_o = generate_features_ranking(x_train_o_rem_sup, y_train_o, '1')
    ranking_kselect_anova_u = generate_features_ranking(x_train_u_rem_sup, y_train_u, '1')
    ranking_kselect_anova_s = generate_features_ranking(x_train_s_rem_sup, y_train_s, '1')

    # display ten best features distribution
    # display_best_feature(x_train, y_train, ranking_kselect_anova, 2)
    # display_best_feature(x_train_u, y_train_u, ranking_kselect_anova_u, 2)
    # display_best_feature(x_train_s, y_train_s, ranking_kselect_anova_s, 2)

    # perform test to find number of best features with which model performs the best
    ranking_accuracy_rates = []
    numbers_of_features = list(range(2, x_train_o_rem_sup.shape[1], 2))
    ranking_accuracy_rates.append(bayes_n_features(x_train_o_rem_sup, x_test[x_train_o_rem_sup.columns], y_train_o, y_test, labels, ranking_kselect_anova_o, numbers_of_features))
    ranking_accuracy_rates.append(bayes_n_features(x_train_u_rem_sup, x_test[x_train_u_rem_sup.columns], y_train_u, y_test, labels, ranking_kselect_anova_u, numbers_of_features))
    ranking_accuracy_rates.append(bayes_n_features(x_train_s_rem_sup, x_test[x_train_s_rem_sup.columns], y_train_s, y_test, labels, ranking_kselect_anova_s, numbers_of_features))

    #plot_rankings(numbers_of_features, ranking_accuracy_rates)

    #unsupervised selection
    data.pop('class')
    columns_within_groups = unsupervised_feature_selection(data, 0.1)

    ############### DISCRETIZATION #########################
    columns_to_skip = ['class']

    # on underbalanced data
    #dis_w_x_train = discretize_width(x_train_u, 5, columns_to_skip)

    #dis_d_x_train = discretize_depth(x_train, 5, columns_to_skip)

    ############### DUMMYFICATION - sn2 dataset #############
    #dummied_sub_frames = []
    # groups are between [0, 56, 140, 322, 755] - for each chose 20 best features
    #pca = PCA(n_components=20)
    #!!!!
    # for group in columns_within_groups:
    #     #principalComponents = pca.fit_transform(dis_w_x_train[group])
    #     kowalski = 25
    #     if len(group) >100:
    #         kowalski = 40
    #     dummylist = []
    #     for att in dis_w_x_train[group[:kowalski]]:
    #         if att in ['a01', 'a02']: dis_w_x_train[att] = dis_w_x_train[att].astype('category')
    #         dummylist.append(pd.get_dummies(dis_w_x_train[[att]]))
    #     dummified_x_train = pd.concat(dummylist, axis=1)
    #     dummied_sub_frames.append(dummified_x_train)
    #     dummified_x_train.head(5)

    ############### ASSOCIATIONS RULE #####################

    # get best columns unsupervised
    best_columns = []
    for x in columns_within_groups:
        best_columns.append(x[:len(x)//10])
    best_columns_flat = []
    best_columns_flat = [item for sublist in best_columns for item in sublist]
    best_columns = best_columns_flat
    # dummylist = []
    # for att in dis_w_x_train[best_columns]:
    #     if att in ['a01', 'a02']: dis_w_x_train[att] = dis_w_x_train[att].astype('category')
    #     dummylist.append(pd.get_dummies(dis_w_x_train[[att]]))
    # dummified_x_train = pd.concat(dummylist, axis=1)
    #
    # dummied_sub_frames = [dummified_x_train]
    #
    # association_rules_list = []
    # nr_of_patterns = []
    # avg_quality_rules_confidence = []
    # avg_quality_top_rules = []
    # avg_quality_rules_lift =[]
    # min_sup_list =[]
    # nr_of_patterns_avg =[]
    # for sub_data in dummied_sub_frames:
    #     frequent_itemsets = {}
    #     minpaterns = 50
    #     minsup = 0.85
    #     while minsup < 1:
    #
    #         frequent_itemsets = apriori(sub_data, min_support=minsup, use_colnames=True)
    #
    #         minconf = 0.7
    #         try:
    #             rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    #         except:
    #             pass
    #         min_sup_list.append(minsup)
    #         rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    #         nr_of_patterns.append(len(frequent_itemsets))
    #         nr_of_patterns_avg.append(rules["antecedent_len"].median())
    #         avg_quality_rules_confidence.append(rules["confidence"].median())
    #         avg_quality_rules_lift.append(rules["lift"].median())
    #         avg_quality_top_rules.append(rules["confidence"][(rules['antecedent_len'] >= 2)].median())
    #         minsup += 0.04
    #         print(rules["antecedent_len"].value_counts())
    #         if len(frequent_itemsets) >= minpaterns:
    #             print("Minimum support:", minsup)
    #
    #         association_rules_list.append(frequent_itemsets)
    #     print("Number of found patterns:", len(frequent_itemsets))
    #
    #     print(nr_of_patterns)
    #     print(avg_quality_rules_confidence)
    #     print(avg_quality_top_rules)
    #     print(avg_quality_rules_lift)
    #
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(min_sup_list, avg_quality_rules_confidence)
    #     plt.plot(min_sup_list, avg_quality_top_rules)
    #     plt.plot(min_sup_list, avg_quality_rules_lift)
    #     #plt.plot(min_sup_list, nr_of_patterns_avg)
    #
    #     plt.ylabel('level of rule measures')
    #     plt.xlabel('minimum support')
    #     plt.legend(labels=['avg rules confidence', 'avg conf best rules', 'avg rules lift'])
    #     plt.grid()
    #     plt.show()

        # rulesx = association_rules(association_rules_list[-1], metric="confidence", min_threshold=minconf)
        # rulesx["antecedent_len"] = rulesx["antecedents"].apply(lambda x: len(x))
        # print(rulesx[(rulesx['antecedent_len'] >= 3)])
        #
        # rulesx.sort_values(['confidence'], axis=1, ascending=False, inplace=True)
        # print(rulesx.iloc[:,0:2])
        #




    # for item in association_rules_list:
    #     rules = association_rules(item, metric="confidence", min_threshold=minconf)
    #     rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    #     print(rules[(rules['antecedent_len'] >= 3)])
    #
    # rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
    # rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
    # best_attributes =[]
    # list_best_patter = pd.unique(rules['consequents'][(rules['antecedent_len'] >= 2)])
    # for x in list_best_patter:
    #     if x not in best_attributes:
    #         best_attributes.append(x)
    #
    # list_best_patter = pd.unique(rules['antecedents'][(rules['antecedent_len'] >= 2)])
    # for x in list_best_patter:
    #     if x not in best_attributes:
    #         best_attributes.append(x)
    #
    # print(best_attributes)
    ############### CLUSTERING ##########################

    #
    # kf =  KFold(n_splits=7)
    # #kf.get_n_splits(trnX)
    #
    # tree = DecisionTreeClassifier(max_depth=4)
    # for train_index_tree, test_index_tree in kf.split(x_train_o, y=y_train_o):
    #     #print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = x_train_o[train_index_tree], x_train_o[test_index_tree]
    #     y_train, y_test = y_train_o[train_index_tree], y_train_o[test_index_tree]
    #
    #     tree.fit(X_train, y_train)
    #
    # tree.fit(x_test, y_test)
    #
    # file_out_name_png = 'dtree1.png'
    # dot_data = export_graphviz(tree, out_file='dtree.dot', filled=True, rounded=True, special_characters=True)
    # # Convert to png
    #
    # call(['dot', '-Tpng', 'dtree.dot', '-o', file_out_name_png, '-Gdpi=600'], shell=True)
    # #check_call(['dot', '-Tpng', 'dtree.dot', '-o', 'dtree.png'], shell=True)
    #
    # plt.figure(figsize=(14, 18))
    # plt.imshow(plt.imread(file_out_name_png))
    # plt.axis('off')
    # plt.show()

    # sum_sq_distances = []
    # slihouettes = []
    # calinsis_hrab = []
    # davies_buldins = []
    # silhouete_per_instances = []
    #
    # sh_euc = []
    # sh_che = []
    # sh_cos = []
    # corell = []
    #
    # adj_rand_score = []
    # clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # for num_clusters in clusters:
    #     # build KMeans model
    #     kmeans_model = cluster.KMeans(n_clusters=num_clusters, random_state=1).fit(x_train_o[best_columns])
    #     y_pred = kmeans_model.labels_
    #
    #     sum_sq_distances.append(kmeans_model.inertia_)
    #     # print("Sum of squared distances:", sum_sq_distances[-1])
    #
    #     slihouettes.append(metrics.silhouette_score(x_train_o[best_columns], y_pred, metric='euclidean'))
    #     # print("Silhouette:", slihouettes[-1])
    #
    #     calinsis_hrab.append(metrics.calinski_harabasz_score(x_train_o[best_columns], y_pred))
    #     # print("Calinski Harabaz:", calinsis_hrab[-1])
    #
    #     davies_buldins.append(metrics.davies_bouldin_score(x_train_o[best_columns], y_pred))
    #     # print("Davies Bouldin:", davies_buldins[-1])
    #
    #     silhouete_per_instances.append(metrics.silhouette_samples(x_train_o[best_columns], y_pred))
    #     # print("Silhouette per instance:\n", silhouete_per_instances[-1])
    #
    #     sh_euc.append(metrics.silhouette_score(x_train_o[best_columns], y_pred, metric='euclidean'))
    #     # print("Silhouette[blobs with Euclidean] =", sh_euc[-1])
    #     sh_che.append(metrics.silhouette_score(x_train_o[best_columns], y_pred, metric='chebyshev'))
    #     # print("Silhouette[blobs with Chebyshev] =", sh_che[-1])
    #     sh_cos.append(metrics.silhouette_score(x_train_o[best_columns], y_pred, metric='cosine'))
    #     # print("Silhouette[blobs with Cosine] =", sh_cos[-1])
    #     corell.append(metrics.silhouette_score(x_train_o[best_columns], y_pred, metric='correlation'))
    #     # print("Silhouette[blobs with Correlation] =", corell[-1])
    #
    #     adj_rand_score.append(metrics.adjusted_rand_score(y_train_o, y_pred))
    #     # print("RI[incorrect #blobs] =", adj_rand_score[-1])
    #
    # plt.figure(figsize=(20, 4))
    # # subplot 1
    # plt.subplot(141)
    # plt.plot(clusters, sum_sq_distances)
    # plt.ylabel('sum square distance')
    # plt.xlabel('number of clusters')
    # plt.grid()
    # plt.title("Sum of squared distances")
    #
    # # subplot 2
    # plt.subplot(142)
    # plt.plot(clusters, slihouettes)
    # plt.ylabel('slihouettes')
    # plt.xlabel('number of clusters')
    # plt.grid()
    # plt.title("Silhouette:")
    #
    # # subplot 3
    # plt.subplot(143)
    # plt.plot(clusters, calinsis_hrab)
    # plt.ylabel('Calinski Harabaz:')
    # plt.xlabel('number of clusters')
    # plt.grid()
    # plt.title("Calinski Harabaz:")
    #
    # # subplot 4
    # plt.subplot(144)
    # plt.plot(clusters, davies_buldins)
    # plt.ylabel('Davies Bouldin:')
    # plt.xlabel('number of clusters')
    # plt.grid()
    # plt.title("Davies Bouldin:")
    # plt.show()
    #
    # plt.figure(figsize=(20, 4))
    # for i in range(len(clusters)):
    #     values = silhouete_per_instances[i].tolist()
    #     plt.plot(list(range(len(silhouete_per_instances[0]))), values)
    # plt.legend(clusters)
    # plt.ylabel('sum square distance')
    # plt.xlabel('instance')
    # plt.grid()
    # plt.title("Silhouette per instance:")
    # plt.show()
    #
    # # SILHOUETTES SCORES, DIFFERENT METRICS
    # plt.figure(figsize=(20, 4))
    # # subplot 1
    # plt.subplot(141)
    # plt.plot(clusters, sh_euc)
    # plt.ylabel('eucklidian')
    # plt.xlabel('number of clusters')
    # plt.grid()
    # plt.title("Silhouete score Eucklidians")
    #
    # # subplot 2
    # plt.subplot(142)
    # plt.plot(clusters, sh_che)
    # plt.ylabel('Chebyshew')
    # plt.xlabel('number of clusters')
    # plt.grid()
    # plt.title("Silhouette Chebyshew")
    #
    # # subplot 3
    # plt.subplot(143)
    # plt.plot(clusters, sh_cos)
    # plt.ylabel('Coisine metric:')
    # plt.xlabel('number of clusters')
    # plt.grid()
    # plt.title("Silhouette metrics coisine:")
    #
    # # subplot 4
    # plt.subplot(144)
    # plt.plot(clusters, corell)
    # plt.ylabel('Correletion')
    # plt.xlabel('number of clusters')
    # plt.grid()
    # plt.title("Silhouette correlation")
    # plt.show()
    #
    # plt.show()
    #
    # # adjustend rand score plot
    # plt.figure(figsize=(30, 6))
    # plt.plot(clusters, adj_rand_score)
    # plt.ylabel('Adjusted rand score')
    # plt.xlabel('number of clusters')
    # plt.grid()
    # plt.title("Adjusted rand score")
    # plt.show()
