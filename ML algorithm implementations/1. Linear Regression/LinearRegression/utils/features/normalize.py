import numpy as np


def normalize(features):
    """
    Normalize input features.
    :param features: features to normalize
    :return: normalized features
    """
    features_normalized = np.copy(features).astype(float)
    features_mean = np.mean(features, 0)
    features_deviation = np.std(features, 0)
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # Avoid divide by 0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
