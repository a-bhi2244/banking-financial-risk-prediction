
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Create output folder
output_dir = Path("reports/eda")
output_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Default Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="DEFAULT")
plt.title("Credit Card Default Distribution")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.savefig(output_dir / "default_distribution.png")
plt.close()

# 2. Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="AGE", bins=30, kde=True)
plt.title("Customer Age Distribution")
plt.savefig(output_dir / "age_distribution.png")
plt.close()

# 3. Credit Limit Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="LIMIT_BAL", bins=40, kde=True)
plt.title("Credit Limit Distribution")
plt.savefig(output_dir / "credit_limit_distribution.png")
plt.close()

# 4. Payment Status Analysis
payment_columns = [
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]

payment_data = df[payment_columns].melt(
    var_name="Payment Month",
    value_name="Payment Status"
)

plt.figure(figsize=(10, 5))
sns.countplot(
    data=payment_data,
    x="Payment Status",
    hue="Payment Month"
)
plt.title("Payment Status Analysis")
plt.savefig(output_dir / "payment_status.png")
plt.close()

# 5. Correlation Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm",
    center=0
)
plt.title("Financial Risk Correlation Heatmap")
plt.savefig(output_dir / "correlation_heatmap.png")
plt.close()

print("\n===== EDA COMPLETED =====")
print("All graphs saved in:", output_dir)
print("\nDefault Rate:")
print(df["DEFAULT"].mean() * 100, "%")

print("\nEDA Summary:")
print(df.describe())

