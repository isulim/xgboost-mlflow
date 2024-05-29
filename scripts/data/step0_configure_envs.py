"""Configure environment variables."""

__all__ = ["configure_envs"]


def ask_kaggle_config():
    """Ask for Kaggle config directory."""
    kaggle_dir = input("Enter Kaggle config directory [.kaggle]: ")

    if not kaggle_dir:
        kaggle_dir = ".kaggle"

    return kaggle_dir


def ask_files_dir():
    """Ask for Kaggle files directory."""
    files_dir = input("Enter directory for data files [data]: ")

    if not files_dir:
        files_dir = "data"

    return files_dir


def confirm(files_dir: str, kaggle_dir: str):
    """Confirm input."""

    confirm = input(
        f"""Is provided information correct?
    DATA_FILES_DIR: {files_dir}
    KAGGLE_CONFIG_DIR: {kaggle_dir}
    (Y/n): """
    )
    if not confirm or confirm.lower() == "y":
        with open(".env", "a") as f:
            f.write(f"KAGGLE_FILES_DIR={files_dir}\n")
            f.write(f"KAGGLE_CONFIG_DIR={kaggle_dir}\n")
            f.write(f"KAGGLE_DATASET=meirnizri/covid19-dataset\n")
    else:
        print("Please re-enter the information.")


def configure_envs():
    kaggle_dir = ask_kaggle_config()
    files_dir = ask_files_dir()
    confirm(files_dir, kaggle_dir)


if __name__ == "__main__":
    configure_envs()
