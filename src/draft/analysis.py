import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import preprocessing as prep

#################### READ DATA #############################

# import data
data_set_name = '../data/pd_speech_features.csv'
data: pd.DataFrame = pd.read_csv(data_set_name, index_col='id', sep=',', header=1, decimal='.', parse_dates=True,
                                 infer_datetime_format=True)

# print(data.index)
# print(data[data.index == 0])

#################### SPLIT DATA #############################

y = data.pop('class')
x = data
labels = pd.unique(y)
indices = pd.unique(x.index)

# join back data(pop removes column)
data = pd.concat([x, y], axis=1)

# define auxiliary dataframe
aux_df = pd.DataFrame(columns=data.columns)
for index in indices:
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

# print_balance(target_count_dataset, target_count_test, title1='balance of whole set', title2='balance of test set')


x_train, x_test = prep.normalization(x_train, x_test)

#################### NORMALIZE DATA #############################

target_count_dataset = y.replace({0: 'Healthy', 1: 'Sick'}).value_counts()
target_count_test = y_test.replace({0: 'Healthy', 1: 'Sick'}).value_counts()

# print_balance(target_count_dataset, target_count_test, title1='balance of whole set', title2='balance of test set')

x_train, x_test = prep.normalization(x_train, x_test)

#################### DATA BALANCE #############################

# before sampling
target_count = y_train.replace({0: 'Healthy', 1: 'Sick'}).value_counts()
min_class = target_count.idxmin()
ind_min_class = target_count.index.get_loc(min_class)
# auxiliary dictionary to plot data
values = {'Original': [target_count['Healthy'], target_count['Sick']]}

# SMOTE
x_train_s, y_train_s = prep.balance_with_SMOTE(x_train, y_train)

# undersampling
x_train_u, y_train_u = prep.perform_balance_undersampling(x_train, y_train, plot_data=True)

# oversamppling
x_train_o, y_train_o = prep.balance_oversampling(x_train, y_train)

minor = y_train_s.replace({0: 'Healthy', 1: 'Sick'}).value_counts()

values['SMOTE'] = [minor['Healthy'], minor['Sick']]

minor = y_train.replace({0: 'Healthy', 1: 'Sick'}).value_counts()
values['OVERSAMPLING'] = [minor['Healthy'], minor['Sick']]

minor = y_train_u.replace({0: 'Healthy', 1: 'Sick'}).value_counts()
values['UNDERSAMPLING'] = [minor['Healthy'], minor['Sick']]

# print(values)
# display_balance_result(values)

#################### CORRELATED DATA #############################

x_train_o_rem_sup = prep.correlation_removal(x_train_o)
x_train_u_rem_sup = prep.correlation_removal(x_train_u)
x_train_s_rem_sup = prep.correlation_removal(x_train_s)

#################### FEATURE SELECTION #############################

ranking_kselect_anova_o = prep.generate_features_ranking(x_train_o_rem_sup, y_train_o, '1')
ranking_kselect_anova_u = prep.generate_features_ranking(x_train_u_rem_sup, y_train_u, '1')
ranking_kselect_anova_s = prep.generate_features_ranking(x_train_s_rem_sup, y_train_s, '1')

# display ten best features distribution
# display_best_feature(x_train, y_train, ranking_kselect_anova, 2)
# display_best_feature(x_train_u, y_train_u, ranking_kselect_anova_u, 2)
# display_best_feature(x_train_s, y_train_s, ranking_kselect_anova_s, 2)


# unsupervised selection
data.pop('class')
columns_within_groups = prep.unsupervised_feature_selection(data, 0.1)

############### DISCRETIZATION #########################
columns_to_skip = ['class']

# on underbalanced data
# dis_w_x_train = discretize_width(x_train_u, 5, columns_to_skip)

# dis_d_x_train = discretize_depth(x_train, 5, columns_to_skip)

############### ASSOCIATIONS RULE #####################

# get best columns unsupervised
best_columns = []
for x in columns_within_groups:
    best_columns.append(x[:len(x) // 10])
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

