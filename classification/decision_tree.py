import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left_tree=None, right_tree=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left_tree = left_tree
        self.right_tree = right_tree
        self.value = value
    
    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, cf):
        self.max_depth = cf["max_depth"]
        self.min_split = cf["min_split"]
        self.root = None
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_on_tree(x, self.root) for x in X])

    def _predict_on_tree(self, x, node: Node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_on_tree(x, node.left_tree)
        else:
            return self._predict_on_tree(x, node.right_tree) 

    def _build_tree(self, X, y, d=0):
        rows, features = X.shape
        classes = len(np.unique(y))

        if d >= self.max_depth or rows < self.min_split or classes == 1:
            val = Counter(y).most_common(1)[0][0]
            return Node(value=val)
        
        best_f, best_t = self._split(X, y)

        if best_f is None:
            val = Counter(y).most_common(1)[0][0]
            return Node(value=val)
        
        L = X[:, best_f] <= best_t
        R = ~L

        left_tree = self._build_tree(X[L], y[L], d + 1)
        right_tree = self._build_tree(X[R], y[R], d + 1)

        return Node(feature=best_f, threshold=best_t, left_tree=left_tree, right_tree=right_tree)
    
    #перебирая все возможные уникальные значения в наборе находим лучушую фичу и её порог
    def _split(self, X, y):
        best_gain = -1
        best_f, best_t = None, None

        for f_ind in range(X.shape[1]):
            thresholds = np.unique(X[:, f_ind])
            for t in thresholds:
                gain = self._split_factor(X, y, f_ind, t)
                if gain > best_gain:
                    best_gain = gain
                    best_f = f_ind
                    best_t = t
        
        return best_f, best_t
    
    #используем вычитание из всего набора, поскольку наша цель найти прирост полезной информации.
    #если прирост случается, то двигаемся к следующему порогу
    def _split_factor(self, X, y, feature_ind, threshold):
        total_gini = self._gini(y)
        L = X[:, feature_ind] <= threshold
        R = ~L

        if len(y[L]) == 0 or len(y[R]) == 0:
            return 0
        
        Ln = len(y[L]) / len(y)
        Rn = len(y[R]) / len(y)

        return total_gini - (Ln * self._gini(y[L]) + Rn * self._gini(y[R]))  

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p**2)


if __name__ == "__main__":
    pass