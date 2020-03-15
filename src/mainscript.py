import pandas as pd
import numpy as np
import decision_tree as dtree
import bayes
import random_forest as rforest
import instance_based as knn
from pandas.plotting import register_matplotlib_converters
import xg_boost


if __name__ == '__main__':
    # READ DATA SET
    register_matplotlib_converters()

    data_set_name = '../data/pd_speech_features.csv'
    data = pd.read_csv(data_set_name, index_col='id',sep=',', header=1, decimal='.', parse_dates=True,
                       infer_datetime_format=True)

    groups = pd.read_csv(data_set_name, sep=',', nrows=1, decimal='.', parse_dates=True,
                       infer_datetime_format=True)

    groups_header = groups.dtypes

    y = data.pop('class')
    x = data

    bayes_model = bayes.NaiveBayes(5)
    result_bayes = bayes_model.naive_bayes(x, y, 20)
    print('Result for naive Bayes', result_bayes)

    # result_knn = knn.investigate_KNN(x, y, 20)
    # print('Result for KNN', result_knn)
    #
    # result_tree = dtree.investigate_decision_tree(x, y, 20)
    # print('Result for KNN', result_tree)
    #
    # result_forest = rforest.investigate_random_forest(x, y, 20)
    # print('Result for KNN', result_forest)
    #
    # result_xgboost = xg_boost.investigate_xg_boost(x, y, 20)
    # print('Result for XGBoost', result_xgboost)
