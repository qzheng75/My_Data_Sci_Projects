import numpy as np


def get_dataset():
    mu = np.array([1, 2, 3])
    covariance = np.array([[2, 0.5, 0.7],
                           [0.5, 3, 0.9],
                           [0.7, 0.9, 4]])

    # Generate a 3D dataset with 1000 samples
    size = 1000
    dataset = np.random.multivariate_normal(mu, covariance, size=size)
    return dataset


class PCA:
    def __init__(self):
        self.cumulative_explained_variance = None
        self.eig_vec = None
        self.mean = None

    def fit_pca(self, data, n_components):
        mean = np.mean(data, axis=0)
        self.mean = mean
        mean_data = data - mean
        covariance_mat = np.cov(mean_data.T)
        eig_val, eig_vec = np.linalg.eig(covariance_mat)

        # Sort all eigenvectors
        idx = np.arange(0, len(eig_val))
        idx = ([x for _, x in sorted(zip(eig_val, idx))])[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]

        # Choose only n_components
        eig_val = eig_val[: n_components]
        eig_vec = eig_vec[:, : n_components]
        self.eig_vec = eig_vec

        sum_eig_val = np.sum(eig_val)
        explained_variance = eig_val / sum_eig_val
        cumulative_explained_variance = np.cumsum(explained_variance)
        self.cumulative_explained_variance = cumulative_explained_variance

        # Project the data
        pca_data = np.dot(mean_data, eig_vec)
        return pca_data

    def inverse_transform(self, pca_data):
        return np.dot(pca_data, self.eig_vec.T) + self.mean


if __name__ == '__main__':
    data = get_dataset()
    model = PCA()
    pca_data = model.fit_pca(data, 2)
    reconstruct_data = model.inverse_transform(pca_data)
    print(np.mean(np.square(reconstruct_data - data)))
