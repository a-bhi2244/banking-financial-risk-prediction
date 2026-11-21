
import pandas as pd
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "default_credit_card.xls"
)

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_excel(
        DATA_PATH,
        sheet_name="Data",
        header=1
    )

    df.columns = df.columns.astype(str).str.strip()

    df.rename(
        columns={"default payment next month": "DEFAULT"},
        inplace=True
    )

    return df


if __name__ == "__main__":
    df = load_dataset()

    print("\n===== DATASET INFORMATION =====")
    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    print("\nDefault Distribution:")
    print(df["DEFAULT"].value_counts())

    print("\n===== DATASET LOADED SUCCESSFULLY =====")
