import os

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

__all__ = ["split_train_val_test"]

from sklearn.model_selection import train_test_split


def split_train_val_test(x: pd.DataFrame, y: pd.DataFrame, size: float = 0.1, seed: int = 42):
    """Split data to train, validation and test sets."""

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=size, random_state=seed)

    return x_train, y_train, x_val, y_val, x_test, y_test


def save_to_csv(x: pd.DataFrame, y: pd.DataFrame, out: Path) -> None:
    """Save X, y dataframes to CSV."""

    if not out.exists():
        os.makedirs(out)
    print(f"Saving X and y to CSV in: {out}")

    x.to_csv(Path(out, "X.csv"), index=False)
    y.to_csv(Path(out, "y.csv"), index=False)


if __name__ == "__main__":
    load_dotenv()
    root_data = os.getenv("KAGGLE_FILES_DIR")
    dataset_path = Path(os.getcwd(), root_data)
    X = pd.read_csv(Path(dataset_path, "preprocessed", "X_pre.csv"))
    y = pd.read_csv(Path(dataset_path, "preprocessed", "y_pre.csv"))
    split_train_val_test(X, y, size=0.1, seed=42)
