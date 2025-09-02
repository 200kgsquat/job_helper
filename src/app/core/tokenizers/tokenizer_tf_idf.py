# src/app/core/tokenizers/tokenizer_tf_idf.py
"""
Combined text cleaner and module-level tokenizer for TF-IDF pipeline.

This file contains:
- TextCleaner: an sklearn-compatible transformer used to clean text
- tokenizer_func: a top-level function that joblib can import when unpickling
  pipelines that were saved with tokenizer=tokenizer_func.

Make sure this file path (module path) is the same one used when the pipeline
was originally saved.
"""

import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk.stem import WordNetLemmatizer

# Ensure nltk resources are available (downloads are quiet)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Text cleaning transformer compatible with sklearn Pipeline.
    Methods intentionally mirror your original implementation.
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def remove_urls(self, text: str) -> str:
        return re.sub(r"#URL_[A-Za-z0-9]+#", "", text)

    def remove_stopwords(self, text: str) -> str:
        return " ".join([w for w in text.split() if w.lower() not in ENGLISH_STOP_WORDS])

    def lemmatize(self, text: str) -> str:
        return " ".join([self.lemmatizer.lemmatize(w) for w in text.split()])

    def handle_punctuation(self, text: str) -> str:
        return " ".join(re.findall(r"\b\w+\b", text.lower()))
    
    def handle_empty(self, text):
        """Handle empty/NaN values for individual text entries"""
        return "" if pd.isna(text) else str(text)
    
    def clean_text(self, text) -> str:
        """Full cleaning pipeline for a single text string."""
        text = self.handle_empty(text)
        text = text.lower()
        text = self.remove_urls(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        text = self.handle_punctuation(text)
        return text.strip()

    # sklearn compatibility
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is an iterable of raw text strings
        return [self.clean_text(x) for x in X]


# Create one module-level instance to be used by tokenizer_func.
_text_cleaner = TextCleaner()


def tokenizer_func(text):
    """
    Module-level tokenizer function used by TfidfVectorizer(tokenizer=...).

    IMPORTANT:
    - This must be a top-level function (not nested) so joblib/pickle can import it
      using the module path used when the pipeline was saved.
    - It should return a list of tokens. The function below cleans the text
      then splits on whitespace. If your original pipeline expected another
      behavior (e.g. returning a string), adapt this to match the original.
    """
    cleaned = _text_cleaner.clean_text(text)
    # Return list of tokens (tokenizer for TfidfVectorizer expects tokens)
    return cleaned.split()
