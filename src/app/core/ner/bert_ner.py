from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    AutoConfig,
    pipeline,
)
from datasets import Dataset
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Union
import os, random, csv
from seqeval.metrics import f1_score, classification_report

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

    class CustomTrainer(Trainer):
        """Trainer subclass with optional focal loss."""
        def __init__(self, *args, use_focal=False, focal_gamma=2.0, focal_alpha=0.25, **kwargs):
            super().__init__(*args, **kwargs)
            self.use_focal = use_focal
            self.focal_gamma = focal_gamma
            self.focal_alpha = focal_alpha

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            if self.use_focal and labels is not None:
                num_labels = logits.shape[-1]
                logits_flat = logits.view(-1, num_labels)
                labels_flat = labels.view(-1)
                ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction='none', ignore_index=-100)
                pt = torch.exp(-ce_loss)
                focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
                mask = labels_flat != -100
                loss = focal_loss[mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=logits.device, requires_grad=True)
                return (loss, outputs) if return_outputs else loss
            else:
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
                return (loss, outputs) if return_outputs else loss

    @staticmethod
    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, id2label: Dict[int, str]):
        preds = np.argmax(predictions, axis=-1)
        batch_size, seq_len = label_ids.shape
        preds_list, labels_list = [], []
        for i in range(batch_size):
            pred_seq, label_seq = [], []
            for j in range(seq_len):
                if label_ids[i, j] == -100:
                    continue
                label_seq.append(id2label[int(label_ids[i, j])])
                pred_seq.append(id2label[int(preds[i, j])])
            preds_list.append(pred_seq)
            labels_list.append(label_seq)
        return preds_list, labels_list

    def train(
        self,
        train_texts: List[List[str]],
        train_labels: List[List[str]],
        val_texts: List[List[str]] = None,
        val_labels: List[List[str]] = None,
        output_dir="./models/bert-ner",
        epochs=3,
        batch_size=16,
        learning_rate=5e-5,
        gradient_accumulation_steps=1,
        label_all_tokens=False,
        use_focal=False,
        focal_gamma=2.0,
        focal_alpha=0.25,
        oversample_entities=1,
        seed=42,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.makedirs(output_dir, exist_ok=True)

        # Oversampling
        if oversample_entities > 1:
            new_texts, new_labels = [], []
            for t, l in zip(train_texts, train_labels):
                new_texts.append(t)
                new_labels.append(l)
                if any(lbl != "O" for lbl in l):
                    for _ in range(oversample_entities - 1):
                        new_texts.append(t)
                        new_labels.append(l)
            train_texts, train_labels = new_texts, new_labels

        # Label setup
        if not self.label_list:
            self.label_list = sorted({lab for labs in train_labels for lab in labs})
            self.id2label = {i: l for i, l in enumerate(self.label_list)}
            self.label2id = {l: i for i, l in self.id2label.items()}

            config = self.model.config
            config.id2label = self.id2label
            config.label2id = self.label2id
            config.num_labels = len(self.label_list)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                config=config,
                ignore_mismatched_sizes=True
            )

        # Datasets
        train_dataset = Dataset.from_dict({"tokens": train_texts, "labels": train_labels})
        val_dataset = Dataset.from_dict({"tokens": val_texts, "labels": val_labels}) if val_texts else None

        def tokenize_and_align_labels(examples):
            tokenized = self.tokenizer(examples["tokens"], is_split_into_words=True, truncation=True)
            all_labels = []
            for i, labels in enumerate(examples["labels"]):
                word_ids = tokenized.word_ids(batch_index=i)
                prev_word = None
                label_ids_seq = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids_seq.append(-100)
                    elif word_idx != prev_word:
                        label_ids_seq.append(self.label2id[labels[word_idx]])
                    else:
                        label_ids_seq.append(self.label2id[labels[word_idx]] if label_all_tokens else -100)
                    prev_word = word_idx
                all_labels.append(label_ids_seq)
            tokenized["labels"] = all_labels
            return tokenized

        tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["tokens", "labels"])
        tokenized_val = val_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["tokens", "labels"]) if val_dataset else None

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            weight_decay=0.01,
            save_strategy="no",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=os.path.join(output_dir, "logs"),
            log_level="info",
            seed=seed,
            fp16=torch.cuda.is_available(),
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        def compute_metrics(p):
            preds_list, labels_list = self.align_predictions(p.predictions, p.label_ids, self.id2label)
            return {"f1": f1_score(labels_list, preds_list)}

        trainer = self.CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics if val_dataset else None,
            use_focal=use_focal,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )

        trainer.train()

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Export errors
        if val_dataset:
            predictions, label_ids, _ = trainer.predict(tokenized_val)
            preds_list, labels_list = self.align_predictions(predictions, label_ids, self.id2label)
            errors_path = os.path.join(output_dir, "errors.csv")
            with open(errors_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["sentence", "true_labels", "pred_labels"])
                for words, true_l, pred_l in zip(val_texts, labels_list, preds_list):
                    if true_l != pred_l:
                        writer.writerow([" ".join(words), " ".join(true_l), " ".join(pred_l)])
            print(f"F1-score (seqeval): {f1_score(labels_list, preds_list):.4f}")
            print("\nClassification report:\n")
            print(classification_report(labels_list, preds_list))

        return trainer

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
