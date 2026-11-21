
import pandas as pd
from pathlib import Path
from src.data_loader import load_dataset


def clean_dataset(df):

    print("\n===== DATA CLEANING STARTED =====")

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove ID because it is only a customer identifier
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Handle missing numerical values
    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].median())

    # Handle missing categorical values
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


if __name__ == "__main__":

    df = load_dataset()

    print("Original Dataset Shape:", df.shape)

    cleaned_df = clean_dataset(df)

    output_path = Path("data/processed/cleaned_data.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(output_path, index=False)

    print("\nCleaned Dataset Shape:", cleaned_df.shape)
    print("\nMissing Values After Cleaning:")
    print(cleaned_df.isnull().sum().sum())

    print("\nTarget Distribution:")
    print(cleaned_df["DEFAULT"].value_counts())

    print("\nCleaned dataset saved at:", output_path)
    print("\n===== DATA CLEANING COMPLETED =====")
