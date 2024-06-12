configure-envs:
	poetry run python scripts/data/configure_envs.py

download-data:
	poetry run python scripts/data/download_data.py

preprocess-data:
	poetry run python scripts/data/preprocessing.py

mlflow-start:
	docker compose up -d mlflow-srv

mlflow-stop:
	docker compose down mlflow-srv

mlflow-clear:
	docker compose up -d
	docker compose exec mlflow-srv mlflow gc
	docker compose down

experiment:
	poetry run python main.py
