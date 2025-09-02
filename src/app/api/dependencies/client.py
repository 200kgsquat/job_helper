# src/app/api/dependencies/client.py
import os
import joblib
import sys
import types

# Import tokenizer module and expose tokenizer_func as __main__.tokenizer_func so pickle can resolve it
from src.app.core.tokenizers import tokenizer_tf_idf as _tk_module

_main_mod = sys.modules.get("__main__")
if _main_mod is None:
    _main_mod = types.ModuleType("__main__")
    sys.modules["__main__"] = _main_mod
setattr(_main_mod, "tokenizer_func", getattr(_tk_module, "tokenizer_func"))

# Now safe to load the joblib pipeline (it will find __main__.tokenizer_func)
TFIDF_MODEL_PATH = os.path.join("models", "tfidf_logreg_pipeline.joblib")
tfidf_pipeline = joblib.load(TFIDF_MODEL_PATH)

# wrap it into your wrapper if needed
from src.app.core.classifiers.tf_idf import TFIDFWrapper
tfidf_model = TFIDFWrapper()
tfidf_model.pipeline = tfidf_pipeline

# load BERT and NER as before...
from src.app.core.classifiers.bert import BertWrapper
from src.app.core.ner.bert_ner import BertNER

bert_model = BertWrapper()
try:
    bert_model.load(os.path.join("models", "bert-wrapper-model"))
except Exception:
    pass

ner_model = BertNER()
try:
    ner_model.load(os.path.join("models", "bert-ner-skillspan"))
except Exception:
    pass

def get_tfidf_model():
    return tfidf_model

def get_bert_model():
    return bert_model

def get_ner_model():
    return ner_model
