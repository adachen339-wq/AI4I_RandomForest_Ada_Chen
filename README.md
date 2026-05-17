# AI4I_RandomForest_Ada_Chen
Machine Failure Prediction using Custom Random Forest
## Overview
This project implements a **Random Forest algorithm from scratch** to predict machine failure using industrial sensor data.

Unlike standard implementations, this project builds the ensemble model manually using bootstrap sampling and decision tree aggregation.

## Key Features
- Custom Random Forest implementation (no sklearn RF used)
- Hyperparameter tuning (grid search)
- Model persistence (pickle)
- Prediction pipeline
- Model interpretability (feature importance)
- Visualization of results

## Dataset
AI4I 2020 Predictive Maintenance Dataset

## Model Architecture
- Base learner: DecisionTreeClassifier
- Ensemble method: Bagging
- Voting: Majority voting

## Results
- Accuracy optimized via grid search
- Feature importance analysis shows key operational variables affecting failure

## Visualizations
- Confusion Matrix
- Feature Importance Plot
- True vs Predicted Comparison

## Installation
