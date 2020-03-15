import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import preprocessing as prep
import pandas as pd

#################### READ DATA #############################

# import data
data_set_name = '../data/pd_speech_features.csv'
data: pd.DataFrame = pd.read_csv(data_set_name, index_col='id', sep=',', header=1, decimal='.', parse_dates=True,
                                 infer_datetime_format=True)

y = data.pop('class')
x = data

############## CLUSTERING ##########################

sum_sq_distances = []
slihouettes = []
calinsis_hrab = []
davies_buldins = []
silhouete_per_instances = []

sh_euc = []
sh_che = []
sh_cos = []
corell = []

x_train_o, y_train_o = prep.balance_oversampling(x_train, y_train)
adj_rand_score = []
clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for num_clusters in clusters:
    # build KMeans model
    kmeans_model = cluster.KMeans(n_clusters=num_clusters, random_state=1).fit(x_train_o[best_columns])
    y_pred = kmeans_model.labels_

    sum_sq_distances.append(kmeans_model.inertia_)
    # print("Sum of squared distances:", sum_sq_distances[-1])

    slihouettes.append(metrics.silhouette_score(x_train_o[best_columns], y_pred, metric='euclidean'))
    # print("Silhouette:", slihouettes[-1])

    calinsis_hrab.append(metrics.calinski_harabasz_score(x_train_o[best_columns], y_pred))
    # print("Calinski Harabaz:", calinsis_hrab[-1])

    davies_buldins.append(metrics.davies_bouldin_score(x_train_o[best_columns], y_pred))
    # print("Davies Bouldin:", davies_buldins[-1])

    silhouete_per_instances.append(metrics.silhouette_samples(x_train_o[best_columns], y_pred))
    # print("Silhouette per instance:\n", silhouete_per_instances[-1])

    sh_euc.append(metrics.silhouette_score(x_train_o[best_columns], y_pred, metric='euclidean'))
    # print("Silhouette[blobs with Euclidean] =", sh_euc[-1])
    sh_che.append(metrics.silhouette_score(x_train_o[best_columns], y_pred, metric='chebyshev'))
    # print("Silhouette[blobs with Chebyshev] =", sh_che[-1])
    sh_cos.append(metrics.silhouette_score(x_train_o[best_columns], y_pred, metric='cosine'))
    # print("Silhouette[blobs with Cosine] =", sh_cos[-1])
    corell.append(metrics.silhouette_score(x_train_o[best_columns], y_pred, metric='correlation'))
    # print("Silhouette[blobs with Correlation] =", corell[-1])

    adj_rand_score.append(metrics.adjusted_rand_score(y_train_o, y_pred))
    # print("RI[incorrect #blobs] =", adj_rand_score[-1])