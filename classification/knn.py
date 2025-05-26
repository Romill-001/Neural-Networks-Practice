import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.X = None
        self.y = None
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    # убрать корень и квадрать попробовать найти среднее квадратное отклонение
    def _calculate_dist(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _predict_on_single(self, x):
        dists = np.array([self._calculate_dist(x, x_tr) for x_tr in self.X])
        knn_indices = dists.argsort()[:self.k]
        knn_labels = self.y[knn_indices]
        return Counter(knn_labels).most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self._predict_on_single(x) for x in X])