import csv
from datetime import datetime
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

pca = PCA(2)
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


results_db_index = {}
results_c_index = {}
results_dunn_index = {}
result_labels = {}
result_unique_labels = {}
result_centers = {}
for k in K_values:
    K_means = Deterministic_KMeans(k=k)
    centers = K_means.deterministic_centers(X_train)
    centroids = np.asarray(centers, dtype='int')
    result_centers.update({k: centroids})
    cluster_centers, clustering = K_means.run(centers)

    create_clusters = K_means._create_clusters(centers)
    get_labels = K_means._get_cluster_labels(create_clusters)
    u_labels = np.unique(get_labels)
    u_labels = u_labels.astype(int)
    get_labels = get_labels.astype(int)
    result_unique_labels.update({k: u_labels})
    result_labels.update({k: get_labels})

    # Davies Bouldin index
    db_index = davies_bouldin(X_train, k, clustering, cluster_centers)
    results_db_index.update({k: db_index})

    # C-index
    c_idx = c_index(clustering, X_train)
    results_c_index.update({k: c_idx})

    # Dunn-index
    d_idx = dunn_index(clustering)
    results_dunn_index.update({k: d_idx})

type = ''


def _generate_results_kmeans():
    directory = "../resources/results_kmeans/"
    for kk, labels_values in result_labels.items():
        df = pca.fit_transform(X_train)
        for k in result_unique_labels.keys():
            labels_values = result_labels.get(k)
            values = result_unique_labels.get(k)
            for i in values:
                pos = labels_values == i
                plt.scatter(df[pos, 0], df[pos, 1], label=i)
            plt.legend(bbox_to_anchor=(0.99, 1.0), loc='upper left')
            name_file = 'fig_' + str(k) + '.png'
            plt.savefig(directory + name_file)
            plt.cla()
    return directory


def _generate_clustering_quality():
    directory = "../resources/clustering_quality/"
    list_plot = [{"Davies-Bouldin-index": results_db_index}, {"C-index": results_c_index},
                 {"Dunn-index": results_dunn_index}]
    fig, axs = plt.subplots(3, sharex=False, sharey=False)
    for i, result in enumerate(list_plot):
        name = list(result.keys())
        algorimth = list(result.values())
        axs[i].plot(list(algorimth[0].keys()), list(algorimth[0].values()))
        axs[i].set(xlabel="Number of clusters", ylabel=name[0])

    plt.savefig(directory + "c_q.png")
    return directory



while type != '3':
    print('=== MESSAGE ===')
    print(" If you want to generate the graphs:")
    print(" - Ranking result for each class, you type 1")
    print(" - Clustering quality, you type 2")
    print("If you want to exit, type 3")
    type = input()
    if type == '1':
        directory = _generate_results_kmeans()
        print("     ** The directory where the graphs were save is: ", directory)
        print("")
    elif type == '2':
        directory = _generate_clustering_quality()
        print("     ** The directory where the graph was save is: ", directory)
        print("")
    elif type == '3':
        break
    else:
        print("Invalid order")
        break
print("Finish!")


