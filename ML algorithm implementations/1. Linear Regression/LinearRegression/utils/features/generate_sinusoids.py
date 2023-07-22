import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """
    Generate sinusoid features for input data.
    :param dataset: data to transform
    :param sinusoid_degree: degree for sinusoid features generation
    :return: transformed data
    """
    num_examples = dataset.shape[0]
    sins = np.empty((num_examples, 0))

    for degree in range(1, sinusoid_degree + 1):
        sinusoid_features = np.sin(degree * dataset)
        sins = np.concatenate((sins, sinusoid_features), axis=1)
        
    return sins
