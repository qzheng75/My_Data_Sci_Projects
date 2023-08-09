from .sigmoid import sigmoid


def sigmoid_gradient(matrix):
    return sigmoid(matrix) * (1 - sigmoid(matrix))
