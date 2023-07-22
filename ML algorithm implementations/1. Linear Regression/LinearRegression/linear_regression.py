import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.features.preprocessor import preprocess
import warnings
import logging


class LinearRegressionModel:

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=False):
        """
        Init function for Linear regression model
        """
        (data_processed,
         features_mean,
         features_deviation) = preprocess(data, polynomial_degree, sinusoid_degree, normalize_data)

        self._data = data_processed
        self._labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self._data.shape[1]
        self.w = np.zeros((num_features, 1))

    def __compute_cost(self, regularization_method=None, lambda_=0.):
        m = self._data.shape[0]
        delta = np.dot(self._data, self.w) - self._labels
        cost = 1 / (2 * m) * np.dot(delta.T, delta)
        if regularization_method is not None:
            reg = 0
            if regularization_method == 'l1':
                reg = lambda_ * np.sum(np.abs(self.w))
            if regularization_method == 'l2':
                reg = lambda_ * np.dot(self.w.T, self.w)
            cost += reg
        return cost[0][0]

    def __compute_gradient(self, regularization_method=None, lambda_=0.):
        m = self._data.shape[0]
        delta = np.dot(self._data, self.w) - self._labels
        gradient = 1 / m * np.dot(delta.T, self._data).T
        if regularization_method is not None:
            if regularization_method == 'l1':
                gradient += np.sign(self.w)
            if regularization_method == 'l2':
                gradient += 2 * lambda_ * self.w
        return gradient

    def __gradient_descent(self, alpha):
        gradient = self.__compute_gradient()
        self.w -= alpha * gradient

    def train(self, n_itr=100000, alpha=1e-5, log_cost=True, regularization_method=None, lambda_=0.):
        if regularization_method is not None and regularization_method not in ["l1", "l2"]:
            raise ValueError("Unsupported Regularization method")
        if regularization_method is not None and lambda_ == 0:
            warnings.warn("Regularization selected, yet lambda = 0.")
        verbosity = n_itr / 10
        J_hist = []
        for i in tqdm(range(n_itr)):
            cost = self.__compute_cost(regularization_method=regularization_method,
                                       lambda_=lambda_)
            if i % verbosity == 0 and log_cost:
                print(f"Step {i}: cost: {cost:.4f}")
            if i < 100000:
                J_hist.append(cost)
            self.__gradient_descent(alpha)
        return J_hist

    def predict(self, data):
        data_processed = preprocess(data,
                                    self.polynomial_degree,
                                    self.sinusoid_degree,
                                    self.normalize_data
                                    )[0]
        return np.dot(data_processed, self.w)

    def evaluate(self, X_test, y_test, evaluate_method='r_score'):
        y_pred = self.predict(X_test)
        y_mean = np.mean(y_test)
        ss_regression = np.sum((y_pred - y_test) ** 2)
        ss_total = np.sum((y_mean - y_test) ** 2)
        n = len(y_test)
        if evaluate_method == 'r_score':
            return 1 - ss_regression / ss_total
        elif evaluate_method == 'MSE':
            return 1 / n * ss_regression
        elif evaluate_method == 'RMSE':
            return np.sqrt(1 / n * ss_regression)
        elif evaluate_method == 'MAE':
            return 1 / n * np.sum(np.sign(y_pred - y_test))
        else:
            raise ValueError("Unsupported Evaluation metric")


def plot_linear_fit(w, b, X, y):
    """
    Plot the data and the line fitted by coefficients w and intercept b.
    Only available for uni-variate linear regression.
    """
    plt.figure()
    plt.scatter(X, y, label='Data Points')

    # Calculate the predicted values using the linear model
    y_pred = np.dot(X, w) + b

    # Sort the X values to connect the points in a line
    sorted_indices = np.argsort(X.flatten())
    sorted_X = X[sorted_indices]
    sorted_y_pred = y_pred[sorted_indices]

    # Plot the line fitted by the model
    plt.plot(sorted_X, sorted_y_pred, color='red', label='Linear Fit')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Data and Linear Fit')
    plt.legend()
    plt.grid(True)
    plt.show()
