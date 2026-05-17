# -*- coding: utf-8 -*-
"""
Author: Ada Chen
Project: Machine Failure Prediction using Custom Random Forest
File: train.py
Description: Model training and hyperparameter tuning
Date: 2026
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from AI4I_model_Ada import MyRandomForest


# =========================
# 数据
# =========================

df = pd.read_csv("ai4i2020.csv")

df = df.drop(["UDI", "Product ID"], axis=1)
df = pd.get_dummies(df, columns=["Type"])

X = df.drop("Machine failure", axis=1).values
y = df["Machine failure"].values

feature_names = df.drop("Machine failure", axis=1).columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 调参
# =========================

best_model = None
best_acc = 0
best_params = {}

for n_trees in [10, 20, 30]:
    for depth in [5, 10, 15]:
        for split in [2, 5]:

            model = MyRandomForest(
                n_trees=n_trees,
                max_depth=depth,
                min_samples_split=split
            )

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)

            print(f"trees={n_trees}, depth={depth}, split={split}, acc={acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_params = {
                    "n_trees": n_trees,
                    "max_depth": depth,
                    "min_samples_split": split
                }

print("\nBest Params:", best_params)
print("Best Accuracy:", best_acc)

print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))


# =========================
# 保存模型
# =========================

with open("saved_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nModel saved successfully.")