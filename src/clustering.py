
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Keep target separately for cluster analysis
default_values = df["DEFAULT"].copy()

# Exclude target and ID from clustering
X = df.drop(
    columns=["DEFAULT", "ID"],
    errors="ignore"
)

# Scale features because K-Means is distance-based
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train K-Means
print("Training K-Means Clustering Model...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels and actual default values
results = pd.DataFrame({
    "CLUSTER": clusters,
    "DEFAULT": default_values
})

# Calculate default rate for each cluster
cluster_rates = results.groupby("CLUSTER")["DEFAULT"].mean()

# Sort clusters by observed default rate
sorted_clusters = cluster_rates.sort_values().index.tolist()

# Assign risk labels based on observed default rates
risk_labels = ["Low Risk", "Medium Risk", "High Risk"]

cluster_risk_map = {
    int(cluster): risk_labels[index]
    for index, cluster in enumerate(sorted_clusters)
}

# Add risk labels
results["RISK_SEGMENT"] = results["CLUSTER"].map(
    cluster_risk_map
)

# Display cluster summary
summary = results.groupby(
    ["CLUSTER", "RISK_SEGMENT"]
).agg(
    CUSTOMERS=("DEFAULT", "count"),
    DEFAULT_RATE=("DEFAULT", "mean")
).reset_index()

summary["DEFAULT_RATE"] = (
    summary["DEFAULT_RATE"] * 100
).round(2)

print("\n===== CLUSTERING RESULTS =====")
print(summary.to_string(index=False))

# Calculate silhouette score
score = silhouette_score(
    X_scaled,
    clusters,
    sample_size=3000,
    random_state=42
)

print("\nSilhouette Score:", round(score, 4))

# Save model and metadata
Path("models").mkdir(exist_ok=True)
Path("data/processed").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": kmeans,
        "scaler": scaler,
        "features": X.columns.tolist()
    },
    "models/clustering_model.pkl"
)

joblib.dump(
    {
        "cluster_risk_map": cluster_risk_map,
        "cluster_default_rates": {
            int(k): float(v)
            for k, v in cluster_rates.items()
        }
    },
    "models/clustering_metadata.pkl"
)

# Save customer segmentation results
results.to_csv(
    "data/processed/customer_segments.csv",
    index=False
)

print("\nModels saved successfully!")
print("Customer segments saved successfully!")
print("\n===== CLUSTERING COMPLETED =====")
