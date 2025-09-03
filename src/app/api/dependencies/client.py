from fastapi import Request
from src.app.core.ner.bert_ner import BertNER
from src.app.core.classifiers.tf_idf import TFIDFWrapper

def get_ner_model(request: Request) -> BertNER:
    return request.app.state.ner_model

def get_tfidf_model(request: Request) -> TFIDFWrapper:
    return request.app.state.tfidf_model
