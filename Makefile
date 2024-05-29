configure-envs:
	poetry run python scripts/data/00_configure_envs.py

download-data:
	poetry run python scripts/data/01_download_data.py

preprocess-data:
	poetry run python scripts/data/02_preprocessing.py
