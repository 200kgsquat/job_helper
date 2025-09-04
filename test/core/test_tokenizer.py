# tests/core/test_tokenizer.py
from src.app.core.tokenizers.tokenizer_tf_idf import tokenizer_func, TextCleaner


def test_tokenizer_func_basic():
    s = "Python developer. Visit #URL_12345# now!"
    tokens = tokenizer_func(s)
    assert isinstance(tokens, list)
    assert "python" in [t.lower() for t in tokens]


def test_textcleaner_transform():
    cleaner = TextCleaner()
    texts = ["Hello World!", None, "Python #URL_ABC# and SQL."]
    cleaned = cleaner.transform(texts)
    assert isinstance(cleaned, list)
    assert cleaned[0] != ""
    assert isinstance(cleaned[1], str)
