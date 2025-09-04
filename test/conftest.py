# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from src.app.main import app

# Import the client dependency module so we can monkeypatch its attributes
import src.app.api.dependencies.client as client_module


@pytest.fixture
def client():
    """
    TestClient for the FastAPI app.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_text():
    """
    Single example text used in multiple tests.
    """
    return (
        "I have 5 years of experience as a software engineer. "
        "My skills include Python, FastAPI, SQL, and machine learning. "
        "I have also worked on data pipelines using Apache Airflow. "
        "Contact me at #URL_ABC123# for more details."
    )


class DummyTFIDF:
    """
    Minimal dummy wrapper that mimics the predict API used by your app.
    Returns a constant label for any input.
    """
    def __init__(self, label="DummyLabel"):
        self._label = label
        # keep .pipeline in case some code expects pipeline attribute
        self.pipeline = self

    def predict(self, texts):
        # return a list of labels corresponding to texts length
        return [self._label for _ in texts]


class DummyNER:
    """
    Dummy NER that mimics the predictive interface.
    Returns a list of token-entity dicts (or nested list) consistent with many NER pipelines.
    """
    def predict(self, texts):
        # If single string, return list of entities; if list, return list-of-lists
        def _one(t):
            return [
                {"word": "Python", "label": "SKILL"},
                {"word": "FastAPI", "label": "SKILL"},
                {"word": "SQL", "label": "SKILL"}
            ]
        if isinstance(texts, list):
            return [_one(t) for t in texts]
        else:
            return _one(texts)


@pytest.fixture(autouse=True)
def monkeypatch_models(monkeypatch):
    """
    Automatically monkeypatch the real loaded models in the client dependency
    so tests don't depend on heavyweight actual model files.
    """
    dummy_tfidf = DummyTFIDF()
    dummy_ner = DummyNER()

    # Replace exported variables and getter functions in the client module.
    # This covers code that either calls get_tfidf_model() or reads tfidf_model directly.
    if hasattr(client_module, "tfidf_model"):
        monkeypatch.setattr(client_module, "tfidf_model", dummy_tfidf)
    if hasattr(client_module, "get_tfidf_model"):
        monkeypatch.setattr(client_module, "get_tfidf_model", lambda: dummy_tfidf)

    if hasattr(client_module, "ner_model"):
        monkeypatch.setattr(client_module, "ner_model", dummy_ner)
    if hasattr(client_module, "get_ner_model"):
        monkeypatch.setattr(client_module, "get_ner_model", lambda: dummy_ner)

    # Also patch BERT wrapper getters to avoid unexpected heavy loads (if present).
    if hasattr(client_module, "bert_model"):
        monkeypatch.setattr(client_module, "bert_model", dummy_tfidf)
    if hasattr(client_module, "get_bert_model"):
        monkeypatch.setattr(client_module, "get_bert_model", lambda: dummy_tfidf)

    yield
