# text_cleaner.py
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk.stem import WordNetLemmatizer

# Ensure nltk resources are available
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def remove_urls(self, text):
        return re.sub(r"#URL_[A-Za-z0-9]+#", "", text)

    def remove_stopwords(self, text):
        return " ".join([w for w in text.split() if w.lower() not in ENGLISH_STOP_WORDS])

    def lemmatize(self, text):
        return " ".join([self.lemmatizer.lemmatize(w) for w in text.split()])

    def handle_punctuation(self, text):
        return " ".join(re.findall(r"\b\w+\b", text.lower()))
    
    def handle_empty(self, text):
        """Handle empty/NaN values for individual text entries"""
        return "" if pd.isna(text) else str(text)
    
    def clean_text(self, text):
        # First handle empty/NaN values
        text = self.handle_empty(text)
        text = text.lower()
        text = self.remove_urls(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        text = self.handle_punctuation(text)
        return text.strip()