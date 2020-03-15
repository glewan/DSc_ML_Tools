import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, mutual_info_regression, SelectKBest, mutual_info_classif, \
    f_classif, VarianceThreshold
from sklearn.utils import resample


def perform_data_balance(data, labels, plot_data=False):
    sick = data[labels == 1]
    sick = sick.iloc[::3, :]
    healthy = data[labels == 0]

    # get equivalent number of healthy 'observations'
    healthy = healthy.iloc[:sick.shape[0], :]

    data_new = pd.concat([sick, healthy], axis=0)
    return data_new


def normalization(x_train, x_test):

    indices = x_train.index
    columns = x_train.columns

    # Normalize training data
    std_scale = preprocessing.StandardScaler().fit(x_train)
    # get list of ranges after normalization
    x_train_norm = std_scale.transform(x_train)


    # Transform normalized data to data frame (from ndarray)
    x_train = pd.DataFrame(x_train_norm, index=indices, columns=columns)

    indices = x_test.index
    columns = x_test.columns
    # Normalize testing data
    x_test_norm = std_scale.transform(x_test)
    x_test = pd.DataFrame(x_test_norm, index=indices, columns=columns)

    return x_train, x_test


def generate_features_ranking(x_train, y_train, type):
    if type == '1':
        # apply SelectKBest class to extract top 10 best features
        selector = SelectKBest(score_func=f_classif, k=10)
        x_new = selector.fit_transform(x_train, y_train)
        df_scores = pd.DataFrame(selector.scores_)
        df_columns = pd.DataFrame(x_train.columns)

        # print("Scores:", selector.scores_)
        # concat two dataframes for better visualization
        attributes_scores = pd.concat([df_columns, df_scores], axis=1)
        # name new columns
        attributes_scores.columns = ['Attr', 'Score']
        # print("Original data space:\n", x[0:3], "\nNew data space:\n", X_new[0:3])
        # print("ANOVA test based ranking")
        # print(attributesScores.nlargest(10, 'Score'))

        return attributes_scores.sort_values(by='Score', axis=0, ascending=False)

    elif type == '2':
        # apply SelectPercantiles class to extract top 10 best features
        selector = SelectPercentile(f_classif, percentile=50)
        x_new_percentiles = selector.fit_transform(x_train, y_train)
        df_pvalues = pd.DataFrame(selector.pvalues_)
        df_columns = pd.DataFrame(x_train.columns)
        # concat two dataframes for better visualization
        attributes_scores_percentile = pd.concat([df_columns, df_pvalues], axis=1)
        # name new columns
        attributes_scores_percentile.columns = ['Attr', 'Pvalues']
        # print("Original data space:\n", x[0:3], "\nNew data space:\n", X_new[0:3])
        print("percantiles test based ranking")
        print(attributes_scores_percentile.nsmallest(10, 'Pvalues'))

        return attributes_scores_percentile.sort_values(by='Pvalues', axis=0)

    elif type == '3':
        # apply SelectKBest class to extract top 10 best features
        selector = SelectKBest(score_func=mutual_info_classif, k=10)
        x_new = selector.fit_transform(x_train, y_train)
        df_scores = pd.DataFrame(selector.scores_)
        df_columns = pd.DataFrame(x_train.columns)
        # print("Scores:", selector.scores_)
        # concat two dataframes for better visualization
        attributes_scores = pd.concat([df_columns, df_scores], axis=1)
        # name new columns
        attributes_scores.columns = ['Attr', 'Score']
        # print("Original data space:\n", x[0:3], "\nNew data space:\n", X_new[0:3])
        print("ANOVA test based ranking")
        print(attributes_scores.nlargest(10, 'Score'))

        return attributes_scores.sort_values(by='Score', axis=0, ascending=False)

    elif type == '4':

        selector = SelectPercentile(mutual_info_regression, percentile=30)
        x_new = selector.fit_transform(x_train, y_train)
        df_scores = pd.DataFrame(selector.scores_)
        df_columns = pd.DataFrame(x_train.columns)
        attributes_scores = pd.concat([df_columns, df_scores], axis=1)
        # name new columns
        attributes_scores.columns = ['Attr', 'Score']
        # print("Original data space:\n", x[0:3], "\nNew data space:\n", X_new[0:3])
        print("ANOVA test based ranking")
        print(attributes_scores.nlargest(10, 'Score'))

        return attributes_scores.sort_values(by='Score', axis=0, ascending=False)
        # ranking = attributesScores_percentile.sort_values(by='Pvalues', axis=0)
    else:
        ValueError("wrong type")


def select_features(x_train, y_train, x_test, n_features, ranking_type='1'):
    ranking = generate_features_ranking(x_train, y_train, ranking_type)
    attributes = ranking.head(n_features)

    return x_train[attributes['Attr']], x_test[attributes['Attr']]


def split_back(dataset):
    y_train = dataset.pop('class')
    x_train = dataset

    return x_train, y_train


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


def perform_balance_undersampling(train_x, train_y, plot_data=False):

    data = pd.concat([train_x, train_y], axis = 1)
    # get class balance before performing balance
    target_count_before = data.replace(to_replace={'class': {0: 'Healthy', 1: 'Sick'}})['class'].value_counts()

    # undersample majority class - keep only one observation per sick patient
    sick = data[data['class'] == 1]
    sick = sick.iloc[::3, :]
    healthy = data[data['class'] == 0]
    # get equivalent number of healthy 'observations'
    healthy = healthy.iloc[:sick.shape[0], :]

    data_new = pd.concat([sick, healthy], axis=0)

    target_count_after = data_new.replace(to_replace={'class': {0: 'Healthy', 1: 'Sick'}})['class'].value_counts()
    if plot_data:
        # print_balance(target_count_before, target_count_after, title1='Before undersampling', title2='Class balance after')
        pass

    return split_back(data_new)


def balance_with_SMOTE(x_train, y_train):
    data = pd.concat([x_train, y_train], axis=1)
    target_count = data['class'].value_counts()
    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)
    smote = SMOTE(ratio='minority')

    smote_x, smote_y = smote.fit_sample(x_train, y_train)
    x = pd.DataFrame(smote_x, columns=x_train.columns)
    x['class'] = smote_y

    return split_back(x)


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


def unsupervised_feature_selection(data, treshhold):

    groups_parkinson = [0, 56, 140, 322, 755]
    columns_within_groups = []
    for i in range(len(groups_parkinson) - 1):
        columns_to_analyze = data.columns[groups_parkinson[i]:groups_parkinson[i + 1]]
        x = data[columns_to_analyze]
        # print("Number of features =", X.shape[1])
        selector = VarianceThreshold()
        selector.fit(x)
        values = selector.variances_
        zipped = zip(values, x.columns)
        sorted_zip = sorted(zipped, key=lambda tup: tup[0])


        # print("Feature variance =", values)
        columns_to_append = []
        for x in sorted_zip:
            if x[0] >= treshhold:
                columns_to_append.append(x[1])

        # selector = VarianceThreshold(threshold= treshhold)
        # X_new = selector.fit_transform(X)

        # print("Number of features =", X_new.shape[1])
        columns_within_groups.append(columns_to_append)

    num_columns_removed = data.shape[1]
    for x in columns_within_groups:
        num_columns_removed -= len(x)

    print("Number of columns removed in unsupervised approach", num_columns_removed)

    return columns_within_groups


def select_single_probes(data, labels, freq):
    single_indices = [i for i in range(0, len(labels), freq)]
    single_data = data.iloc[single_indices]
    single_labels = labels.iloc[single_indices]
    return single_data, single_labels
