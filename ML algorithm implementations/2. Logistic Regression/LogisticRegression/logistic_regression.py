import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import warnings
from utils.features.preprocessor import preprocess


class LogisticRegressionModel:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=False):
        """
        Init function for Linear regression model
        """
        (data_processed,
         features_mean,
         features_deviation) = preprocess(data, polynomial_degree, sinusoid_degree, normalize_data)

        self.data = data_processed
        self._onehot_encoder = OneHotEncoder(sparse_output=False)
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        self.original_labels = np.unique(labels)
        self.labels = self._onehot_encoder.fit_transform(labels.reshape(-1, 1))
        num_features = self.data.shape[1]
        num_labels = self.labels.shape[1]
        self.w = np.zeros((num_features, num_labels))

    def __compute_loss(self):
        W = self.w
        X = self.data
        Y = self.labels
        Z = - X @ W
        m = X.shape[0]
        # Trace: correct class probabilities for the corresponding samples in X.
        # Softmax term: actual softmax probabilities for samples in X
        return 1 / m * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))

    def __compute_gradient(self, regularization_method, lambda_):
        W = self.w
        X = self.data
        Y = self.labels
        Z = - X @ W
        P = softmax(Z, axis=1)
        m = X.shape[0]
        gradient = 1 / m * (X.T @ (Y - P))
        if regularization_method == 'l1':
            gradient += lambda_ * np.sign(W)
        elif regularization_method == 'l2':
            gradient += 2 * lambda_ * W
        return gradient

    def __gradient_descent(self, max_iter, alpha, log_loss, regularization_method, lambda_):
        if max_iter > 100000:
            warnings.warn("Too many iterations. Cost after 100000 iterations are not recorded.")

        steps = []
        J_hist = []
        verbosity = max_iter / 10

        for i in range(max_iter):
            cost = self.__compute_loss()
            if log_loss and i % verbosity == 0:
                print(f"Cost at iteration {i}: {cost:.4f}")
            if i <= 100000:
                steps.append(i)
                J_hist.append(cost)
            gradient = self.__compute_gradient(regularization_method, lambda_)
            self.w -= alpha * gradient

        return steps, J_hist

    def train(self, max_iter=1000, alpha=0.1, log_loss=True, regularization_method="", lambda_=0.):
        if regularization_method != "":
            if regularization_method not in ["l1", "l2"]:
                raise ValueError("Unsupported regularization method.")
            if lambda_ == 0:
                warnings.warn("You select a regularizaton method, yet the regularization strength is 0.")
        return self.__gradient_descent(max_iter, alpha, log_loss, regularization_method, lambda_)

    def predict(self, X):
        X, _, _ = preprocess(X)
        Z = -X @ self.w
        P = softmax(Z, axis=1)
        return self.original_labels[np.argmax(P, axis=1)]

    @staticmethod
    def plot_loss(steps, J_hist):
        df = pd.DataFrame({
            'steps': steps,
            'cost': J_hist
        })
        df.plot(
            x='steps',
            y='cost',
            xlabel='steps',
            ylabel='cost'
        )
        plt.title("Cost vs Iteration plot")
        plt.show()

    def get_confusion_matrix(self, data, labels):
        if not np.array_equal(np.unique(labels), [0, 1]):
            raise ValueError("You can only get confusion matrix for binary classification tasks with labels [0, 1].")
        confusion_mat = np.zeros((2, 2))
        y_pred = self.predict(data)
        y_true = labels
        true_positive = np.sum((y_true == 1) & (y_pred == y_true))
        true_negative = np.sum((y_true == 0) & (y_pred == y_true))
        false_negative = np.sum((y_true == 1) & (y_pred != y_true))
        false_positive = np.sum((y_true == 0) & (y_pred != y_true))
        confusion_mat[0][0], confusion_mat[0][1], confusion_mat[1][0], confusion_mat[1][1] = \
            true_positive, false_positive, false_negative, true_negative
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        return confusion_mat, precision, recall

    def evaluate(self, data, labels, class_labels):
        y_pred = self.predict(data)
        y_true = labels
        num_classes = len(class_labels)
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)

        for i, label in enumerate(class_labels):
            true_positives = np.sum((y_true == label) & (y_pred == label))
            false_positives = np.sum((y_true != label) & (y_pred == label))
            false_negatives = np.sum((y_true == label) & (y_pred != label))

            precision[i] = true_positives / (true_positives + false_positives)
            recall[i] = true_positives / (true_positives + false_negatives)

        return precision, recall

