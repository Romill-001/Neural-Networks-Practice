import numpy as np

class LinearClassifier:
    def __init__(self, learning_rate = 0.01, iters = 1000, lambda_ = 0.01):
        self.learning_rate = learning_rate
        self.iters = iters
        self.lambda_ = lambda_
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        _, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.iters):
            cls = np.dot(X, self.w) + self.b
            y_pred = np.where(cls >= 0, 1, 0)
            error = y_pred - y
            dw = (2 / len(X)) * np.dot(X.T, error) + 2 * self.lambda_ * self.w
            db = (2 / len(X)) * np.sum(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    
    def predict(self, X):
        cls = np.dot(X, self.w) + self.b
        return np.where(cls >= 0, 1, 0)
