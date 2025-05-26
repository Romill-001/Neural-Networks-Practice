import numpy as np
from collections import Counter

# класс ноды, которая содержит в себе ссылки на правую и левую подветки и также показыает
# является ли ветка
class Node:
    def __init__(self, feature=None, feature_name=None, threshold=None, 
                 left_tree=None, right_tree=None, value=None):
        self.feature = feature
        self.feature_name = feature_name
        self.threshold = threshold
        self.left_tree = left_tree
        self.right_tree = right_tree
        self.value = value
    
    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, cf, feature_names=None, mode = 'classification'):
        self.cf = cf if cf is not None else {"max_depth": 10, "min_split": 2}
        self.max_depth = self.cf["max_depth"]
        self.min_split = self.cf["min_split"]
        self.root = None
        self.feature_names = feature_names
        self.mode = mode
    
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

        if d >= self.max_depth or rows < self.min_split or \
        (self.mode == 'classification' and classes == 1):
            if self.mode == 'classification':
                val = Counter(y).most_common(1)[0][0] if len(y) > 0 else 0
            else:
                val = np.mean(y)
            return Node(value=val)
        
        best_f, best_t = self._split(X, y)

        if best_f is None:
            if self.mode == 'classification':
                val = Counter(y).most_common(1)[0][0] if len(y) > 0 else 0
            else:
                val = np.mean(y)
            return Node(value=val)
        
        L = X[:, best_f] <= best_t
        R = ~L

        if np.sum(L) == 0 or np.sum(R) == 0:
            if self.mode == 'classification':
                val = Counter(y).most_common(1)[0][0] if len(y) > 0 else 0
            else:
                val = np.mean(y)
            return Node(value=val)
        
        left_tree = self._build_tree(X[L], y[L], d + 1)
        right_tree = self._build_tree(X[R], y[R], d + 1)

        feature_name = self.feature_names[best_f] if self.feature_names else f"Feature_{best_f}"
        return Node(feature=best_f, feature_name=feature_name, 
                   threshold=best_t, left_tree=left_tree, right_tree=right_tree)
    
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
    
    def _split_factor(self, X, y, feature_ind, threshold):
        if self.mode == 'classification':
            total_facor = self._gini(y)
        else:
            total_facor = self._mse(y)
        L = X[:, feature_ind] <= threshold
        R = ~L

        if len(y[L]) == 0 or len(y[R]) == 0:
            return 0
        
        Ln = len(y[L]) / len(y)
        Rn = len(y[R]) / len(y)

        if self.mode == 'classification':
            return total_facor - (Ln * self._gini(y[L]) + Rn * self._gini(y[R]))
        else:
            return total_facor - (Ln * self._mse(y[L]) + Rn * self._mse(y[R]))

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p**2)
    
    def print_feature_importance(self, node=None, indent=""):
        if node is None:
            node = self.root
        
        if node.is_leaf():
            print(f"{indent}Leaf: class={node.value}")
            return
        
        print(f"{indent}Decision: {node.feature_name} <= {node.threshold:.3f}")
        print(f"{indent}|-- True:")
        self.print_feature_importance(node.left_tree, indent + "|  ")
        print(f"{indent}|__ False:")
        self.print_feature_importance(node.right_tree, indent + "   ")
    
    def get_feature_thresholds(self):
        thresholds = {}
        
        def traverse(node):
            if node is None or node.is_leaf():
                return
            
            if node.feature_name not in thresholds:
                thresholds[node.feature_name] = []
            thresholds[node.feature_name].append(node.threshold)
            
            traverse(node.left_tree)
            traverse(node.right_tree)
        
        traverse(self.root)
        return thresholds
    
    def _mse(self, y):
        if len(y) == 0:
            return 0
        m = np.mean(y)
        return np.mean((y - m) ** 2)