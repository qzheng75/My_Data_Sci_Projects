import numpy as np

class SVM:
    def __init__(self, X, y, learning_rate=1e-3, lambda_param=1e-2, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        m, n = X.shape
        self.data = X
        self.labels = y
        self.w = np.zeros(n)
        self.b = 0
        self.cls_map = np.where(y <= 0, -1, 1)

    def __compute_constraint(self, idx):
        linear_model = np.dot(self.data[idx], self.w) + self.b
        return self.cls_map[idx] * linear_model >= 1

    def __compute_gradients(self, constrain, idx):
        if constrain:
            dw = self.lambda_param * self.w
            db = 0
            return dw, db
        dw = self.lambda_param * self.w - np.dot(self.cls_map[idx], self.data[idx])
        db = - self.cls_map[idx]
        return dw, db

    def __gradient_descent(self):
        for _ in range(self.n_iters):
            for idx, x in enumerate(self.data):
                constrain = self.__compute_constraint(idx)
                dw, db = self.__compute_gradients(constrain, idx)
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def train(self):
        self.__gradient_descent()

    def predict(self, X):
        estimate = np.dot(X, self.w) + self.b
        prediction = np.sign(estimate)
        return np.where(prediction == -1, 0, 1)


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = make_blobs(
        n_samples=250, n_features=2, centers=2, cluster_std=1.05, random_state=1
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1)

    clf = SVM(X_train, y_train, n_iters=1000)
    clf.train()
    predictions = clf.predict(X_test)

    print("SVM Accuracy:", accuracy_score(y_test, predictions))