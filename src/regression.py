
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Target: historical billing amount
target = "BILL_AMT6"

# Exclude target and customer ID if present
X = df.drop(columns=[target, "DEFAULT", "ID"], errors="ignore")
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training Regression Model...")
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n===== REGRESSION RESULTS =====")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save model and feature names
Path("models").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist()
    },
    "models/regression_model.pkl"
)

print("\nRegression model saved successfully!")
