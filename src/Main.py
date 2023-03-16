import csv
from datetime import datetime
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib

from ClusteringQuality import davies_bouldin, c_index, dunn_index
from Deterministic_KMeans import Deterministic_KMeans

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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


X_train = train[:, 1:]
Y_train = train[:, 0]

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
    K_means = Deterministic_KMeans(k=k)
    centers = K_means.deterministic_centers(X_train)
    cluster_centers, clustering = K_means.run(centers)

    # Davies Boulding index
    db_index = davies_bouldin(X_train, k, clustering, cluster_centers)
    results_db_index.update({k: db_index})

    # C-index
    c_idx = c_index(clustering, X_train)
    results_c_index.update({k: c_idx})

    # Dunn-index
    d_idx = dunn_index(clustering)
    results_dunn_index.update({k: d_idx})


print("Calculating Clustering Quality")

list_plot = [{"Davies-Bouldin-index": results_db_index}, {"C-index": results_c_index}, {"Dunn-index":results_dunn_index}]
fig, axs = plt.subplots(3, sharex=False, sharey=False)
for i, result in enumerate(list_plot):
    name = list(result.keys())
    algorimth = list(result.values())
    axs[i].plot(list(algorimth[0].keys()), list(algorimth[0].values()))
    axs[i].set(xlabel="Number of clusters", ylabel=name[0])

plt.show()
print("Finish!!")

