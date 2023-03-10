import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import pandas as pd
import Distance


class KMeans:

    def __init__(self, k=5, max_iters=100):
        self.K = k
        self.max_iters = max_iters
        # self.plot_steps = plot_steps

        # List of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # The centers (mean vector) for each cluster
        self.centroids = []

    def deterministic_centers(self, x):
        cluster_centers = [np.mean(x, axis=0)]
        for i in range(self.K - 1):
            distances = np.zeros(len(x))
            for id, elem in enumerate(x):
                # If elem is in cluter_centers, we skip it
                if True in [(elem==center).all() for center in cluster_centers]:
                    continue
                closest_distance = np.min([Distance.euclidean(elem, center) for center in cluster_centers])
                distances[id] = closest_distance
            cluster_centers.append(x[np.argmax(distances)])
        return cluster_centers

    def deterministic_predict(self, x):
        clustering_error = 1
        self.X = x
        self.n_samples, self.n_features = x.shape
        while (True):
            old_error = clustering_error
            self.centroids = self.deterministic_centers(x)
            clustering = [[] for j in range(len(self.centroids))]
            for elem in self.X:
                closest_center = np.argmin([Distance.euclidean(elem, center) for center in self.centroids])
                clustering[closest_center].append(elem)

            # Compute new clusters centers and clustering error
            clustering_error = 0
            for cluster_id, cluster in enumerate(clustering):
                assert len(cluster) != 0, f"Cluster {cluster_id} is empty, should restart"
                self.centroids[cluster_id] = np.mean(cluster, axis=0)

                cluster_error = 0
                for elem in cluster:
                    cluster_error += np.dot(np.transpose(elem - self.centroids[cluster_id]), elem - self.centroids[cluster_id])
                clustering_error += cluster_error
            # If the decreasing of the error becomes insignificant we stop (minimum local or global is reached)
            if np.absolute(clustering_error - old_error) < 0.0001 * old_error:
                break

        self.clusters = self._create_clusters(self.centroids)
        return self._get_cluster_labels(self.clusters), clustering, self.centroids

    def random_predict(self, x):
        self.X = x
        self.n_samples, self.n_features = x.shape

        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to the closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            # if self.plot_steps:
            # self.plot()

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            # if self.plot_steps:
            # self.plot()
        clustering = [[] for j in range(len(centroids_old))]
        for elem in self.X:
            closest_center = np.argmin([Distance.euclidean(elem, center) for center in centroids_old])
            clustering[closest_center].append(elem)

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters), clustering, centroids_old

    def _get_cluster_labels(self, clusters):
        # Each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # Distance of the current sample to each centroid
        distances = [Distance.euclidean(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        # Assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # Distances between old and new centroids, for all centroids
        distances = [Distance.euclidean(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

