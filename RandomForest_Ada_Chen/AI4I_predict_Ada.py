# -*- coding: utf-8 -*-
"""
Author: Ada Chen
Project: Machine Failure Prediction using Custom Random Forest
File: predict.py
Description: Load trained model and perform prediction
Date: 2026-05-17
"""

import pandas as pd
import pickle
from AI4I_model_Ada import MyRandomForest


# =========================
# 数据
# =========================

df = pd.read_csv("ai4i2020.csv")

df = df.drop(["UDI", "Product ID"], axis=1)
df = pd.get_dummies(df, columns=["Type"])

X = df.drop("Machine failure", axis=1).values
y = df["Machine failure"].values


# =========================
# 加载模型
# =========================

with open("saved_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded.")


# =========================
# 预测
# =========================

sample = X[:10]
pred = model.predict(sample)

print("Prediction:")
print(pred)