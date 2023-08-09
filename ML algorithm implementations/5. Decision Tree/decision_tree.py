import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, X, y, task, max_depth=100, min_samples_split=2):
        self.X = X
        self.y = y
        self.n_features = None
        self.n_samples = None
        self.n_class_labels = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        if task not in ['regression', 'classification']:
            raise ValueError("Task must be either regression or classification.")
        self.task = task

    def __is_finished(self, depth):
        if depth >= self.max_depth \
                or self.n_class_labels == 1 \
                or self.n_samples < self.min_samples_split:
            return True
        return False

    def __entropy(self, y):
        if self.task == 'classification':
            probs = np.bincount(y) / len(y)
            return - np.sum([p * np.log2(p) for p in probs if p > 0])
        else:
            mean_y = np.mean(y)
            return np.mean((y - mean_y) ** 2)

    def __split_dataset(self, X, threshold):
        left = np.argwhere(X <= threshold).flatten()
        right = np.argwhere(X > threshold).flatten()
        return left, right

    def __information_gain(self, X, y, threshold):
        base_loss = self.__entropy(y)
        left, right = self.__split_dataset(X, threshold)
        n, n_left, n_right = len(y), len(left), len(right)

        if n_left == 0 or n_right == 0:
            return 0
        child_loss = n_left / n * self.__entropy(y[left]) \
            + n_right / n * self.__entropy(y[right])
        # Use C-4.5 algorithm: use rate of information gain instead of pure information gain
        return (base_loss - child_loss) / base_loss

    def __best_split(self, X, y, features):
        best_split = {'score': -1, 'feature': None, 'threshold': None}
        for ft in features:
            X_ft = X[:, ft]
            thresholds = np.unique(X_ft)
            for t in thresholds:
                ig = self.__information_gain(X_ft, y, t)
                if ig > best_split['score']:
                    best_split['score'] = ig
                    best_split['feature'] = ft
                    best_split['threshold'] = t
        return best_split['feature'], best_split['threshold']

    def __build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        if self.__is_finished(depth):
            value = 0
            if self.task == 'classification':
                value = np.argmax(np.bincount(y))
            elif self.task == 'regression':
                value = np.mean(y)
            return Node(value=value)

        random_features = np.random.permutation(self.n_features)
        best_ft, best_threshold = self.__best_split(X, y, random_features)

        left, right = self.__split_dataset(X[:, best_ft], best_threshold)
        left_child = self.__build_tree(X[left, :], y[left], depth + 1)
        right_child = self.__build_tree(X[right, :], y[right], depth + 1)
        return Node(best_ft, best_threshold, left_child, right_child)

    def __traverse_tree(self, x, node: Node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.__traverse_tree(x, node.left)
        return self.__traverse_tree(x, node.right)

    def train(self):
        self.root = self.__build_tree(self.X, self.y)

    def predict(self, X):
        return np.array([self.__traverse_tree(x, self.root) for x in X])

    def evaluate(self, y_true, y_pred):
        if self.task == 'classification':
            accuracy = np.sum(y_true == y_pred) / len(y_true)
            return accuracy
        else:
            score = 1 - np.sum((y_true - y_pred) ** 2) / np.sum(((y_true - y_true.mean()) ** 2))
            return score


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=1
    )
    model = DecisionTree(X, y, task='regression', max_depth=10, min_samples_split=6)
    model.train()
    y_preds = model.predict(X_test)
    print("R:", model.evaluate(y_test, y_preds))

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    clf = DecisionTree(X_train, y_train, task='classification', max_depth=10, min_samples_split=6)
    clf.train()
    y_pred = clf.predict(X_test)
    acc = clf.evaluate(y_test, y_pred)
    print("Accuracy:", acc)

