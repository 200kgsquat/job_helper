import os

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

MODELS_DIR = os.path.join(BASE_DIR, "models")
TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, "tfidf_logreg_pipeline.joblib")
NER_MODEL_PATH = os.path.join(MODELS_DIR, "bert-ner-skillspan")

os.makedirs(MODELS_DIR, exist_ok=True)

if not os.path.exists(TFIDF_MODEL_PATH):
    print(f"Warning: TF-IDF model not found at {TFIDF_MODEL_PATH}")

if not os.path.exists(NER_MODEL_PATH):
    print(f"Warning: NER model not found at {NER_MODEL_PATH}")