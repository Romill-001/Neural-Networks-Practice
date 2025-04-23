import numpy as np

class LinearClassifier:
    def __init__(self, learning_rate = 0.01, iters = 1000, lambda_ = 0.01):
        self.learning_rate = learning_rate
        self.iters = iters
        self.lambda_ = lambda_
        self.w = None
        self.b = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        _ , n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.iters):
            cls = np.dot(X, self.w) + self.b
            y_pred = self._sigmoid(cls)
            error = y_pred - y
            dw = (2 / len(X)) * np.dot(X.T, error) + 2 * self.lambda_ * self.w
            db = (2 / len(X)) * np.sum(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    
    def predict(self, X):
        cls = np.dot(X, self.w) + self.b
        prob = self._sigmoid(cls)
        return np.where(prob >= 0.5, 1, 0)
