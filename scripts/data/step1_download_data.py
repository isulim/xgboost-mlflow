"""Download dataset from Kaggle using Kaggle API."""

import os
import zipfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def download_dataset(ds: str, output_path: str | Path):
    api.authenticate()
    api.dataset_download_files(dataset=ds, path=output_path, quiet=False, unzip=False)


def unzip_dataset(api, ds: str, output_path: str | Path):
    try:
        outfile: str = api.split_dataset_string(ds)[1]
        with zipfile.ZipFile(f"{output_path}/{outfile}.zip") as z:
            print("Unzipping all files...")
            z.extractall(output_path)
        print("Unzipped all files.")

        os.remove(f"{output_path}/{outfile}.zip")
        print("Deleted zip file.")

        os.rename(f"{output_path}/Covid Data.csv", f"{output_path}/covid_data.csv")
        print(f"Output CSV stored in: {output_path}/covid_data.csv")

    except zipfile.BadZipFile as e:
        raise ValueError(
            f"Bad zip file, please report on www.github.com/kaggle/kaggle-api, details: {e}",
        )
    except FileNotFoundError as e:
        print(e)


def data_download(raw_path: Path, dataset_name: str | None = None):
    os.makedirs(raw_path, exist_ok=True)

    if not dataset_name:
        dataset_name: str = os.getenv("KAGGLE_DATASET", "")

    from kaggle import api  # type: ignore

    print("Download start...")
    download_dataset(dataset_name, raw_path)
    print("Files downloaded successfully.")
    print("Unzipping files...")
    unzip_dataset(api, dataset_name, raw_path)
    print("Files unzipped successfully.")


if __name__ == "__main__":
    kaggle_dir: str = os.getenv("KAGGLE_FILES_DIR", "")
    raw_path: Path = Path(kaggle_dir, "raw")

    data_download(raw_path)
