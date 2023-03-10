import csv
from datetime import datetime
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib

from ClusteringQuality import davies_bouldin, c_index, dunn_index

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os

from sklearn.metrics import accuracy_score

from KMeans import KMeans

dataset_path = '../resources/mnist_small_knn'
path_results = '../resources/resultsQuality'
train_csv = 'train.csv'
test_csv = 'test.csv'


def read_csv(file):
    with open(dataset_path + '/' + file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        matrix = np.array(data, dtype=int)
        return matrix


train = read_csv(train_csv)
# test = read_csv(test_csv)

X_train = train[:, 1:]
Y_train = train[:, 0]
# X_test = test[:, 1:]
# Y_test = test[:, 0:]
labels = np.unique(Y_train)
K_values = [5, 7, 9, 10, 12, 15]


def retrieve_info(cluster_labels, y_train):
    reference_labels = {}
    # For loop to run through each label of cluster label
    for i in range(len(np.unique(cluster_labels))):
        index = np.where(cluster_labels == i, 1, 0)
        inter = y_train[index == 1]
        num = np.bincount(inter).argmax()
        reference_labels[i] = num
    return reference_labels


results_db_index ={}
results_c_index={}
results_dunn_index={}
for k in K_values:
    K_means = KMeans(k=k, max_iters=200)
    y_pred, clustering, centroids = K_means.random_predict(X_train)
    # y_pred, clustering, centroids = K_means.deterministic_predict(X_train)
    # print(len(np.unique(y_pred)))
    # print(retrieve_info(y_pred, Y_train))
    # reference_labels = retrieve_info(y_pred, Y_train)
    # print(reference_labels)
    # number_labels = np.random.rand(len(y_pred))
    # for i in range(len(y_pred)):
        # number_labels[i] = reference_labels[y_pred[i]]

    # Davies Boulding index
    db_index = davies_bouldin(X_train, k, clustering, centroids)
    results_db_index.update({k: db_index})

    # C-index
    c_idx = c_index(clustering, X_train)
    results_c_index.update({k: c_idx})

    # Dunn-index
    d_idx = dunn_index(clustering)
    results_dunn_index.update({k: d_idx})

    # Dunn index
    # dunn_index = dunn_index(X_train, hq)
    # results_dunn_index.update({k:dunn_index})


    # Calculating accuracy score
    # print('value K: ', k, '(%)', accuracy_score(number_labels, Y_train) * 100)

print("Calculating Clustering Quality")

list_plot = [{"Davies-Bouldin-index": results_db_index}, {"C-index": results_c_index}, {"Dunn-index":results_dunn_index}]
fig, axs = plt.subplots(3, sharex=False, sharey=False)
for i, result in enumerate(list_plot):
    name = list(result.keys())
    # print(name)
    algorimth = list(result.values())
    # print(algorimth)
    axs[i].plot(list(algorimth[0].keys()), list(algorimth[0].values()))
    axs[i].set(xlabel="Number of clusters", ylabel=name[0])
    # axs[i].set_title(0)

    # axs[i].xlabel("Number of clusters")
    # plt.show(block=True)
    # plt.interactive(False)
# plt.tight_layout()
plt.show()
print("Finish!!")

