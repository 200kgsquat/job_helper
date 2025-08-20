import re
import pandas as pd
from sklearn.base import TransformerMixin
from transformers import BertTokenizer

class BertTokenizerWrapper(TransformerMixin):
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def remove_urls(self, text):
        return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    def handle_empty(self, text):
        """Handle empty/NaN values for individual text entries"""
        return "" if pd.isna(text) else str(text)

    def preprocess_text(self, text):
        # Handle empty/NaN values
        text = self.handle_empty(text)
        # Remove URLs
        text = self.remove_urls(text)
        return text.strip()

    def transform(self, X, y=None):
        # Preprocess the text
        preprocessed_texts = pd.Series(X).apply(self.preprocess_text)
        # Tokenize the text using BERT tokenizer
        tokenized = preprocessed_texts.apply(
            lambda x: self.tokenizer(
                x,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="np"  # Use NumPy arrays for compatibility with sklearn
            )
        )
        # Extract input IDs and attention masks
        input_ids = [t['input_ids'][0] for t in tokenized]
        attention_masks = [t['attention_mask'][0] for t in tokenized]
        return {"input_ids": input_ids, "attention_mask": attention_masks}