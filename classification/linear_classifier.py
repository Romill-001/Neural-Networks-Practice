import numpy as np

class LinearClassifier:
    def __init__(self, learning_rate = 0.01, iters = 1000, lambda_ = 0.01):
        self.learning_rate = learning_rate
        self.iters = iters
        self.lambda_ = lambda_
        self.w = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _add_bias(self, X):
        return np.column_stack([np.ones(len(X)), X]) 
    
    def fit(self, X, y):
        X = self._add_bias(X)
        s , n_features = X.shape
        self.w = np.zeros(n_features)

        for _ in range(self.iters):
            cls = np.dot(X, self.w)
            y_pred = self._sigmoid(cls)
            error = y_pred - y
            dw = (1 / s) * np.dot(X.T, error)
            dw[1:] += (self.lambda_ / s) * self.w[1:]

            self.w -= self.learning_rate * dw

    def predict(self, X):
        X = self._add_bias(X)
        cls = np.dot(X, self.w)
        prob = self._sigmoid(cls)
        return np.where(prob >= 0.5, 1, 0)