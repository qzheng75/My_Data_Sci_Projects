import numpy as np


class KMeans:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.centroids = self.__init_centroids()

    def __init_centroids(self):
        m = self.data.shape[0]
        random_idx = np.random.permutation(m)
        return self.data[random_idx[:self.k], :]

    def __find_closet_centroids(self):
        m = self.data.shape[0]
        closest_centroid_ids = np.zeros((m, 1))
        for i in range(m):
            dist = np.zeros((self.k, 1))
            for j in range(self.k):
                dist[j] = np.sum((self.data[i] - self.centroids[j]) ** 2)
            closest_centroid_ids[i] = np.argmin(dist)
        return closest_centroid_ids

    def __update_centroids(self, closest_centroid_id):
        n = self.data.shape[1]
        centroids = np.zeros((self.k, n))
        for i in range(self.k):
            idx = closest_centroid_id == i
            centroids[i] = np.mean(self.data[idx.flatten(), :], axis=0)
        self.centroids = centroids

    def train(self, n_itr=100):
        m = self.data.shape[0]
        closest_centroid_ids = np.zeros((m, 1))
        for i in range(n_itr):
            closest_centroid_ids = self.__find_closet_centroids()
            self.__update_centroids(closest_centroid_ids)
        return self.centroids, closest_centroid_ids
