import numpy as np
import Distance
from sklearn.metrics import pairwise_distances


def davies_bouldin(X, n_cluster, cluster_k, centroids):
    # print('=== Calculating Davies Bouldin ===')
    # calculate cluster dispersion
    S = [np.mean([Distance.euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
    Ri = []

    for i in range(n_cluster):
        Rij = []
        # establish similarity between each cluster and all other clusters
        for j in range(n_cluster):
            if j != i:
                r = (S[i] + S[j]) / Distance.euclidean(centroids[i], centroids[j])
                Rij.append(r)
        # select Ri value of most similar cluster
        Ri.append(max(Rij))

    # get mean of all Ri values
    dbi = np.mean(Ri)

    return dbi


def c_index(clustering, train):
    # print('=== Calculating C-Index ===')
    total_distance = 0
    n_pairs = 0
    for cluster_id, cluster in enumerate(clustering):
        distance_matrix = pairwise_distances(cluster, cluster, metric='euclidean')
        # Keep only elements above the diagonal because of symmetry
        distance_matrix = np.triu(distance_matrix)
        cluster_sum = np.sum(distance_matrix)
        total_distance += cluster_sum

        n = len(distance_matrix)
        # Number of elements in upper diagonal of the matrix: sum from 1 to n-1
        n_pairs += ((n - 1) * n) / 2

    # Need to have an integer for slicing below
    n_pairs = int(n_pairs)
    assert isinstance(n_pairs, int), "Should have an integer"

    all_distances = pairwise_distances(train, train, metric='euclidean')
    # Keep only elements above the diagonal because of symmetry, then flatten and remove 0's
    all_distances = np.triu(all_distances)
    all_distances = all_distances.reshape(-1)
    all_distances = all_distances[all_distances != 0]
    assert 0 not in all_distances, "Error while computing distances"
    all_distances.sort()

    min = np.sum(all_distances[:n_pairs])
    max = np.sum(all_distances[-n_pairs:])
    assert len(all_distances[:n_pairs]) == n_pairs
    assert len(all_distances[-n_pairs:]) == n_pairs

    return (total_distance - min) / (max - min)


def dunn_index(clustering):
    # print('=== Calculating Dunn-Index ===')
    # Compute diameter max
    diameters = []
    for cluster in (clustering):
        distance_matrix = pairwise_distances(cluster, cluster, metric='euclidean')
        diameters.append(np.max(distance_matrix))
    diameter_max = np.max(diameters)

    clusters_distances = []
    for cluster1_id, cluster1 in enumerate((clustering)):
        for cluster2 in clustering[cluster1_id + 1:]:
            distance_matrix = pairwise_distances(cluster1, cluster2, metric='euclidean')
            # Single linkage -> min
            clusters_distances.append(np.min(distance_matrix))

    return np.min(clusters_distances) / diameter_max
