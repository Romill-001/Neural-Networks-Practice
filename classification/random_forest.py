import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from decision_tree import DecisionTree

class RandomForestClassifier:
    def __init__(self, num_trees=10, num_features="auto", cf=None, mode = 'classification'):
        self.num_trees = num_trees
        self.num_features = num_features
        self.cf = cf if cf is not None else {"max_depth": 10, "min_split": 2}
        self.trees = []
        self.feature_subsets = []
        self.feature_names = None
        self.mode = mode

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = list(range(X.shape[1]))
            
        n_features = X.shape[1]
        
        if self.num_features == "auto":
            self.num_features = int(np.sqrt(n_features))
        elif self.num_features == "sqrt":
            self.num_features = int(np.sqrt(n_features))
        elif self.num_features == "log2":
            self.num_features = int(np.log2(n_features))
        elif isinstance(self.num_features, int):
            pass
        else:
            raise ValueError("num_features должно быть int, 'auto', 'sqrt' или 'log2'")

        self.feature_subsets = [
            np.random.choice(n_features, size=self.num_features, replace=False)
            for _ in range(self.num_trees)
        ]
        
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        self.trees = Parallel(n_jobs=-1)(
            delayed(self._train_tree)(X_np, y_np, cols)
            for cols in self.feature_subsets
        )

    def _train_tree(self, X, y, feature_indices):
        X_subset = X[:, feature_indices]
        tree = DecisionTree(self.cf,mode=self.mode)
        tree.fit(X_subset, y)
        return tree

    def predict(self, X):
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        tree_preds = np.array([
            tree.predict(X_np[:, cols])
            for tree, cols in zip(self.trees, self.feature_subsets)
        ])
        if self.mode == 'classification':
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)
        else:
            return np.mean(tree_preds, axis=0)