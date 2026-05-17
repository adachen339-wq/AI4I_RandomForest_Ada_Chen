# -*- coding: utf-8 -*-
"""
Author: Ada Chen
Project: Machine Failure Prediction using Custom Random Forest
File: model.py
Description: Custom Random Forest model implementation
Date: 2026
"""

import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier


class MyRandomForest:
    def __init__(self, n_trees=20, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees):
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[idxs], y[idxs]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        final_preds = []
        for preds in tree_preds.T:
            final_preds.append(Counter(preds).most_common(1)[0][0])

        return np.array(final_preds)

    def feature_importance(self):
        return np.mean(
            [tree.feature_importances_ for tree in self.trees],
            axis=0
        )