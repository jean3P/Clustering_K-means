import numpy as np
import Distance
from datetime import datetime

class Deterministic_KMeans:
    def __init__(self, k):
        self.k = k

    def deterministic_centers(self, train_ds):
        self.train_ds = train_ds
        cluster_centers = [np.mean(self.train_ds, axis=0)]
        for i in range(self.k - 1):
            distances = np.zeros(len(self.train_ds))
            for id, elem in enumerate(self.train_ds):
                # If elem is in cluster_centers, we skip it
                if True in [(elem == center).all() for center in cluster_centers]:
                    continue
                closest_distance = np.min([Distance.euclidean(elem, center) for center in cluster_centers])
                distances[id] = closest_distance
            cluster_centers.append(train_ds[np.argmax(distances)])
        return cluster_centers

    def run(self, cluster_centers):
        start_time = datetime.now()
        clustering_error = 1
        while (True):
            first = False
            old_error = clustering_error
            # Empty clustering at each iteration
            clustering = [[] for j in range(len(cluster_centers))]
            for elem in self.train_ds:
                closest_center = np.argmin([Distance.euclidean(elem, center) for center in cluster_centers])
                clustering[closest_center].append(elem)

            # Compute new clusters centers and clustering error
            clustering_error = 0
            for cluster_id, cluster in enumerate(clustering):
                assert len(cluster) != 0, f"Cluster {cluster_id} is empty, should restart"
                cluster_centers[cluster_id] = np.mean(cluster, axis=0)

                cluster_error = 0
                for elem in cluster:
                    cluster_error += np.dot(np.transpose(elem - cluster_centers[cluster_id]),
                                            elem - cluster_centers[cluster_id])
                clustering_error += cluster_error

            print('.', end='', flush=True)
            # If the decreasing of the error becomes insignificant we stop (minimum local or global is reached)
            if np.absolute(clustering_error - old_error) < 0.0001 * old_error:
                end_time = datetime.now()
                print(" Finish K_means: {}".format(self.k), ', duration: ', (end_time-start_time).total_seconds() ,"s.\n")
                break

        return cluster_centers, clustering

    def _get_cluster_labels(self, clusters):
        # Each sample will get the label of the cluster it was assigned to
        self.n_samples, self.n_features = self.train_ds.shape
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
         # Assign the samples to the closest centroids
         clusters = [[] for _ in range(self.k)]
         for idx, sample in enumerate(self.train_ds):
             centroid_idx = self._closest_centroid(sample, centroids)
             clusters[centroid_idx].append(idx)
         return clusters

    def _closest_centroid(self, sample, centroids):
        # Distance of the current sample to each centroid
        distances = [Distance.euclidean(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx