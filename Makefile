run:
	uv run uvicorn src.app.main:app --reload

run_etl:
	uv run python -m src.app.scripts.etl.run_etl

train_tf_idf:
	uv run python -m src.app.scripts.classifiers.train_train_tf_idf

train-bert:
	uv run python -m src.app.scripts.classifiers.train_bert
