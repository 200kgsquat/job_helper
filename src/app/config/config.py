import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

CLEANED_DATA_FILE = os.path.join(DATA_DIR, 'cleaned_job_postings.csv')

TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, 'tfidf_logreg_pipeline.joblib')
BERT_MODEL_PATH = os.path.join(MODELS_DIR, 'bert_model')  # папка с токенайзером и весами

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
