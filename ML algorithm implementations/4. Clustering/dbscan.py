import numpy as np
import matplotlib.pyplot as plt


class MyDBSCAN:
    def __init__(self, data, minPts, radius):
        self.data = data
        self.minPts = minPts
        self.radius = radius

    def __get_neighbor_points(self, data_idx):
        center = self.data[data_idx]
        nbr_points = []
        m = self.data.shape[0]
        for i in range(m):
            if np.linalg.norm(self.data[i] - center) < self.radius:
                nbr_points.append(i)
        return nbr_points

    def __grow_cluster(self, labels, seed_point, nbrs, cluster_label):
        labels[seed_point] = cluster_label
        i = 0
        while i < len(nbrs):
            nbr = nbrs[i]
            if labels[nbr] == -1:
                labels[nbr] = cluster_label
            elif labels[nbr] == 0:
                labels[nbr] = cluster_label
                new_nbr = self.__get_neighbor_points(nbr)
                if len(new_nbr) >= self.minPts:
                    nbrs += new_nbr
            i += 1

    def train(self):
        m = self.data.shape[0]
        labels = [0] * m
        cluster_label = 0
        for P in range(m):
            if labels[P] != 0:
                continue

            nbr_P = self.__get_neighbor_points(P)
            if len(nbr_P) < self.minPts:
                labels[P] = -1
            else:
                cluster_label += 1
                self.__grow_cluster(labels, P, nbr_P, cluster_label)
        return labels, cluster_label

    def plot_res(self, cluster_res, cluster_num):
        m = self.data.shape[0]
        scatter_colors = ['black', 'green', 'brown', 'red', 'purple', 'orange', 'yellow']
        for i in range(cluster_num):
            if i == 0:
                color = 'blue'
            else:
                color = scatter_colors[i % len(scatter_colors)]
            x1 = []
            y1 = []
            for j in range(m):
                if cluster_res[j] == i:
                    x1.append(self.data[j, 0])
                    y1.append(self.data[j, 1])
            plt.scatter(x1, y1, c=color, alpha=1, marker='.')
        plt.title("Clustering result for MyDBSCAN")


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    # Create three gaussian blobs to use as our clustering data.
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)

    X = StandardScaler().fit_transform(X)

    ###############################################################################
    # My implementation of DBSCAN
    #

    # Run my DBSCAN implementation.
    print('Running my implementation...')
    model = MyDBSCAN(X, 10, 0.3)
    labels, num_labels = model.train()
    model.plot_res(labels, num_labels)
    plt.show()

    ###############################################################################
    # Scikit-learn implementation of DBSCAN
    #

    print('Runing scikit-learn implementation...')
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    skl_labels = db.labels_

    # Scikit learn uses -1 to for NOISE, and starts cluster labeling at 0. I start
    # numbering at 1, so increment the skl cluster numbers by 1.
    for i in range(0, len(skl_labels)):
        if not skl_labels[i] == -1:
            skl_labels[i] += 1

    ###############################################################################
    # Did we get the same results?

    num_disagree = 0

    # Go through each label and make sure they match (print the labels if they do not)
    for i in range(0, len(skl_labels)):
        if not skl_labels[i] == labels[i]:
            print('Scikit learn:', skl_labels[i], 'mine:', labels[i])
            num_disagree += 1

    if num_disagree == 0:
        print('PASS - All labels match!')
    else:
        print('FAIL -', num_disagree, 'labels don\'t match.')
