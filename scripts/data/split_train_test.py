import os

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

__all__ = ["split_train_val_test"]

from sklearn.model_selection import train_test_split


def split_train_val_test(x: pd.DataFrame, y: pd.DataFrame, split_size: float = 0.1, seed: int = 42):
    """Split data to train, validation and test sets."""

    print("Splitting data to train, validation and test sets.")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_size, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=split_size, random_state=seed)

    return x_train, y_train, x_val, y_val, x_test, y_test


def save_to_csv(x: pd.DataFrame, y: pd.DataFrame, out: Path) -> None:
    """Save X, y dataframes to CSV."""

    if not out.exists():
        os.makedirs(out)
    print(f"Saving X and y to CSV in: {out}")

    x.to_csv(Path(out, "X.csv"), index=False)
    y.to_csv(Path(out, "y.csv"), index=False)


def split_and_save(base_data_path: Path, split_size: float = 0.1, seed: int = 42) -> None:
    """Split datasets and save to CSV files."""

    x = pd.read_csv(Path(base_data_path, "preprocessed", "X_pre.csv"))
    y = pd.read_csv(Path(base_data_path, "preprocessed", "y_pre.csv"))
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(x, y, split_size=split_size, seed=seed)

    train_path = Path(base_data_path, "train")
    val_path = Path(base_data_path, "val")
    test_path = Path(base_data_path, "test")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    save_to_csv(x_train, y_train, train_path)
    save_to_csv(x_val, y_val, val_path)
    save_to_csv(x_test, y_test, test_path)


if __name__ == "__main__":
    load_dotenv()
    root_data = os.getenv("KAGGLE_FILES_DIR")
    dataset_path = Path(os.getcwd(), root_data)
    split_and_save(dataset_path)
