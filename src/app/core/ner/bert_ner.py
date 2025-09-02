from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    AutoConfig,
)
import torch
import os
import json
from typing import List, Union, Dict

class BertNER:
    def __init__(self, model_name="dslim/bert-base-NER", label_list=None, device=None):
        self.model_name = model_name
        self.label_list = label_list

        if device is None:
            if torch.cuda.is_available():
                self.device = 0
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = -1
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if label_list:
            self.id2label = {i: l for i, l in enumerate(label_list)}
            self.label2id = {l: i for i, l in enumerate(label_list)}
            config = AutoConfig.from_pretrained(
                model_name,
                num_labels=len(label_list),
                id2label=self.id2label,
                label2id=self.label2id
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True
            )
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id

        self.ner_pipeline = None

    def predict(self, texts: Union[str, List[str]]):
        if not self.ner_pipeline:
            self.ner_pipeline = pipeline(
                "token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                aggregation_strategy="simple"
            )
        if isinstance(texts, list):
            return [self.ner_pipeline(" ".join(txt) if isinstance(txt, list) else txt) for txt in texts]
        return self.ner_pipeline(texts)

    def load(self, path: str):
        """Load model, tokenizer, and label list."""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        label_path = os.path.join(path, "label_list.json")
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                self.label_list = json.load(f)
            self.id2label = {i: l for i, l in enumerate(self.label_list)}
            self.label2id = {l: i for i, l in enumerate(self.label_list)}

        self.ner_pipeline = None
        print(f"✅ NER model loaded from {path}")