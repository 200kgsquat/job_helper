import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CLEANED_DATA_FILE = os.path.join(DATA_DIR, 'cleaned_job_postings.csv')

TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, 'tfidf_logreg_pipeline.joblib')
if not os.path.exists(TFIDF_MODEL_PATH):
    print(f"Warning: TF-IDF model not found at {TFIDF_MODEL_PATH}")

NER_MODEL_PATH = os.path.join(MODELS_DIR, 'bert-ner-skillspan')
if not os.path.exists(NER_MODEL_PATH):
    print(f"Warning: NER model not found at {NER_MODEL_PATH}")