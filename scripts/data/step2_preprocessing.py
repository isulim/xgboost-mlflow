import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

__all__ = ["preprocess_data"]

load_dotenv()
root_data = os.getenv("KAGGLE_FILES_DIR")
dataset_path = Path(os.getcwd(), root_data)


def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing values (other that 1 or 2) in relevant columns"""

    print("Dropping missing values in relevant columns.")

    # 9999-99-99 for missing patient date died, i.e. patient lived.
    df["DEATH"] = [1 if each == "9999-99-99" else 2 for each in df.DATE_DIED]

    for col in [
        "PNEUMONIA",
        "DIABETES",
        "HIPERTENSION",
        "RENAL_CHRONIC",
    ]:
        df = df[(df[col] == 1) | (df[col] == 2)]

    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant columns."""

    print("Dropping irrelevant columns.")
    irrelevant_cols = [
        "SEX",
        "PREGNANT",
        "COPD",
        "ASTHMA",
        "INMSUPR",
        "OTHER_DISEASE",
        "TOBACCO",
        "OBESITY",
        "CARDIOVASCULAR",
        "MEDICAL_UNIT",
        "INTUBED",
        "ICU",
        "DATE_DIED",
    ]

    df.drop(columns=irrelevant_cols, inplace=True)

    return df


def map_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Map binary values as 0-1"""
    print("Mapping binary values to 0-1")
    df["CLASIFFICATION_FINAL"] = [1 if clsf > 3 else 2 for clsf in df.CLASIFFICATION_FINAL]

    cols = df.columns.copy()
    cols = cols.drop("AGE")
    for col in cols:
        df[col] = [0 if row == 1 else 1 for row in df[col]]

    return df


def rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to less medical terms."""

    print("Renaming columns.")

    df["COVID"] = df["CLASIFFICATION_FINAL"]
    df["HOSPITALIZED"] = df["PATIENT_TYPE"]
    df["RENAL"] = df["RENAL_CHRONIC"]
    df["MEDICAL_CARE"] = df["USMER"]

    df.drop(
        ["CLASIFFICATION_FINAL", "PATIENT_TYPE", "RENAL_CHRONIC", "USMER"],
        axis=1,
        inplace=True,
    )
    return df


def save_to_csv(df: pd.DataFrame, out: Path) -> None:
    """Save X, y dataframes to CSV."""

    if not out.exists():
        os.makedirs(out)
    print(f"Saving X and y to CSV in: {out}")
    y = df["DEATH"]
    X = df.drop(["DEATH"], axis=1)
    X.to_csv(Path(out, "X_pre.csv"), index=False)
    y.to_csv(Path(out, "y_pre.csv"), index=False)


def preprocess_data(dataset_path: Path) -> None:
    raw_path = Path(dataset_path, "raw")
    print(f"Reading data from dataset path: {raw_path}")
    df = pd.read_csv(Path(raw_path, "covid_data.csv"))

    df = drop_missing_values(df)
    df = drop_columns(df)
    df = map_binary(df)
    df = rename_cols(df)

    save_to_csv(df, Path(dataset_path, "preprocessed"))
    print("Preprocessing finished.")


if __name__ == "__main__":
    preprocess_data(Path(dataset_path))
