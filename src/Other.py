import numpy as np
import pandas as pd
import time

from matplotlib import pyplot as plt
from numba import jit
# Use sklearn for a very fast computation of pairwise distances
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


# Loading the training set
# Labels are removed
train_loaded_csv = pd.read_csv("../resources/mnist_small_knn/train.csv")
train_ds = train_loaded_csv.to_numpy()
train_ds = train_ds[:,1:]


# Metrics and validation indices
@jit
def euclidean_distance(v1, v2):
    assert len(v1) == len(v2), "Vectors should have same length"
    return np.sqrt(np.sum((v1 - v2) ** 2))

def dunn_index(clustering):
    # Compute diameter max
    diameters = []
    for cluster in tqdm(clustering):
        distance_matrix = pairwise_distances(cluster, cluster, metric='euclidean')
        diameters.append(np.max(distance_matrix))
    diameter_max = np.max(diameters)

    clusters_distances = []
    for cluster1_id, cluster1 in enumerate(tqdm(clustering)):
        for cluster2 in clustering[cluster1_id+1:]:
            distance_matrix = pairwise_distances(cluster1, cluster2, metric='euclidean')
            # Single linkage -> min
            clusters_distances.append(np.min(distance_matrix))

    return np.min(clusters_distances) / diameter_max

def c_index(clustering):
    total_distance = 0
    n_pairs = 0
    for cluster_id, cluster in enumerate(tqdm(clustering)):
        distance_matrix = pairwise_distances(cluster, cluster, metric='euclidean')
        # Keep only elements above the diagonal because of symmetry
        distance_matrix = np.triu(distance_matrix)
        cluster_sum = np.sum(distance_matrix)
        total_distance += cluster_sum

        n = len(distance_matrix)
        # Number of elements in upper diagonal of the matrix: sum from 1 to n-1
        n_pairs += ((n-1)*n)/2

    # Need to have an integer for slicing below
    n_pairs = int(n_pairs)
    assert isinstance(n_pairs, int), "Should have an integer"

    all_distances = pairwise_distances(train_ds, train_ds, metric='euclidean')
    # Keep only elements above the diagonal because of symmetry, then flatten and remove 0's
    all_distances = np.triu(all_distances)
    all_distances = all_distances.reshape(-1)
    all_distances = all_distances[all_distances != 0]
    assert 0 not in all_distances, "Error while computing distances"
    print("Sorting...")
    all_distances.sort()

    min = np.sum(all_distances[:n_pairs])
    max = np.sum(all_distances[-n_pairs:])
    assert len(all_distances[:n_pairs]) == n_pairs
    assert len(all_distances[-n_pairs:]) == n_pairs

    return (total_distance - min) / (max - min)


# Methods to initialize the cluster centers
# replace=False to not have the same center more than once
def random_centers(k):
    return train_ds[np.random.choice(train_ds.shape[0], k, replace=False), :]

def deterministic_centers(k):
    cluster_centers = [np.mean(train_ds, axis=0)]
    for i in tqdm(range(k-1)):
        distances = np.zeros(len(train_ds))
        for id, elem in enumerate(train_ds):
            # If elem is in cluster_centers, we skip it
            if True in [(elem==center).all() for center in cluster_centers]:
                continue
            closest_distance = np.min([euclidean_distance(elem, center) for center in cluster_centers])
            distances[id] = closest_distance
        cluster_centers.append(train_ds[np.argmax(distances)])
    return cluster_centers


def kmeans(cluster_centers):
    clustering_error = 1
    while(True):
        first = False
        old_error = clustering_error
        # Empty clustering at each iteration
        clustering = [[] for j in range(len(cluster_centers))]
        for elem in train_ds:
            closest_center = np.argmin([euclidean_distance(elem, center) for center in cluster_centers])
            clustering[closest_center].append(elem)

        # Compute new clusters centers and clustering error
        clustering_error = 0
        for cluster_id, cluster in enumerate(clustering):
            assert len(cluster) != 0, f"Cluster {cluster_id} is empty, should restart"
            cluster_centers[cluster_id] = np.mean(cluster, axis=0)

            cluster_error = 0
            for elem in cluster:
                cluster_error += np.dot(np.transpose(elem - cluster_centers[cluster_id]), elem - cluster_centers[cluster_id])
            clustering_error += cluster_error

        print('.', end='', flush=True)
        # If the decreasing of the error becomes insignificant we stop (minimum local or global is reached)
        if np.absolute(clustering_error - old_error) < 0.0001*old_error:
            break

    return cluster_centers, clustering

K_values = [5, 7, 9, 10, 12, 15]
results_c_index={}
results_dunn_index={}
for k in K_values:
    cluster_centers = deterministic_centers(5)
    new_centers, new_clustering = kmeans(cluster_centers)
    score = c_index(new_clustering)
    results_c_index.update({k: score})
    score = dunn_index(new_clustering)
    results_dunn_index.update({k:score})

list_plot = [{"C-index": results_c_index}, {"Dunn-index":results_dunn_index}]
fig, axs = plt.subplots(2, sharex=False, sharey=False)
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

# def main_loop(initialization, scoring, k):
#     print(f"Initialization with {initialization.__name__}")
#     cluster_centers = initialization(k)
#     print(f"Initialization done\n")
#     assert len(cluster_centers) == k, f"Wrong initialization, get {len(cluster_centers)} clusters instead of {k}"
#     print(f"Computing {k}-means")
#     new_centers, new_clustering = kmeans(cluster_centers)
#     # print(new_centers)
#     print(cluster_centers)
#     assert len(new_centers) == k, f"Error in kmeans, get {len(new_centers)} clusters instead of {k}"
#     print(f"\n{k}-means done\n")
#     print(f"Computing {scoring.__name__} score")
#     score = scoring(new_clustering)
#     print(f"{k}-means with {initialization.__name__} initialization and {scoring.__name__} get score of {score}\n")
#
#
# main_loop(deterministic_centers, c_index, 20)
