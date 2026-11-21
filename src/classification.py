
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Target variable
X = df.drop(columns=["DEFAULT", "ID"], errors="ignore")
y = df["DEFAULT"]

# Split data (preserve class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create classifier
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train
print("Training Classification Model...")
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
roc_auc = roc_auc_score(y_test, probabilities)
pr_auc = average_precision_score(y_test, probabilities)

print("\n===== CLASSIFICATION RESULTS =====")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save confusion matrix
Path("reports/eda").mkdir(parents=True, exist_ok=True)

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Default", "Default"],
    yticklabels=["No Default", "Default"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Customer Default Prediction - Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/eda/confusion_matrix.png")
plt.close()

# Save trained model and feature names
Path("models").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist()
    },
    "models/classification_model.pkl"
)

print("\nModel saved: models/classification_model.pkl")
print("Confusion matrix saved: reports/eda/confusion_matrix.png")
print("\n===== CLASSIFICATION COMPLETED =====")
