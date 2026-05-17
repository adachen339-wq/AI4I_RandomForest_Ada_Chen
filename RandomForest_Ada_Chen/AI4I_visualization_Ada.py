# -*- coding: utf-8 -*-
"""
Author: Ada Chen
Project: Machine Failure Prediction using Custom Random Forest
File: visualization.py
Description: Visualization and performance analysis
Date: 2026
"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from AI4I_model_Ada import MyRandomForest


# =========================
# 数据
# =========================

df = pd.read_csv("ai4i2020.csv")

df = df.drop(["UDI", "Product ID"], axis=1)
df = pd.get_dummies(df, columns=["Type"])

feature_names = df.drop("Machine failure", axis=1).columns

X = df.drop("Machine failure", axis=1).values
y = df["Machine failure"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 加载模型
# =========================

with open("saved_model.pkl", "rb") as f:
    model = pickle.load(f)


# =========================
# 预测
# =========================

y_pred = model.predict(X_test)


# =========================
# 图1：混淆矩阵
# =========================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()


# =========================
# 图2：特征重要性
# =========================

importances = model.feature_importance()

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()


# =========================
# 图3：预测对比
# =========================

plt.figure(figsize=(12, 5))
plt.plot(y_test[:100], label="True", marker="o")
plt.plot(y_pred[:100], label="Pred", marker="x")

plt.title("True vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()