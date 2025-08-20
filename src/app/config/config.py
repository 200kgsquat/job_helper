# src/app/config.py
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Data files
CLEANED_DATA_FILE = os.path.join(DATA_DIR, 'cleaned_job_postings.csv')

# Model files
TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, 'tfidf_logreg_pipeline.joblib')
BERT_MODEL_PATH = os.path.join(MODELS_DIR, 'bert_model.pth')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)