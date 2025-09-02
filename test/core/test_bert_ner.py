import pytest
from unittest.mock import patch, MagicMock
from src.app.core.ner.bert_ner import BertNER

@pytest.fixture
def ner_model():
    model = BertNER()
    model.pipeline = MagicMock()
    model.pipeline.return_value = [
        {"entity_group": "PER", "score": 0.99, "word": "John", "start": 11, "end": 15},
        {"entity_group": "ORG", "score": 0.98, "word": "Google", "start": 29, "end": 35}
    ]
    return model

def test_predict_single_text(ner_model):
    text = "My name is John and I work at Google."
    result = ner_model.predict(text)
    assert isinstance(result, list)
    for entity in result:
        assert isinstance(entity, dict)
        keys = {"entity_group", "score", "word", "start", "end"}
        assert keys.issubset(entity.keys())

def test_load_method(tmp_path):
    model_path = tmp_path / "fake_model"
    model_path.mkdir()
    import json
    with open(model_path / "label_list.json", "w", encoding="utf-8") as f:
        json.dump(["O", "PER", "ORG"], f)

    ner = BertNER()
    with patch("src.app.core.ner.bert_ner.AutoTokenizer.from_pretrained"), \
         patch("src.app.core.ner.bert_ner.AutoModelForTokenClassification.from_pretrained"):
        ner.load(str(model_path))
    assert ner.label_list == ["O", "PER", "ORG"]
