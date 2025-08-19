run_etl:
    python src/app/core/etl.py
train_tf_idf:
    python -m src.app.scripts.classifiers.train_tf_idf
train_bert:
    python -m src.app.scripts.classifiers.train_bert
