import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import numpy as np

class BertWrapper:
    def __init__(self, model_name='bert-base-uncased', num_labels=2, batch_size=16, epochs=3, learning_rate=5e-5, test_size=0.2, random_state=42):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.label_encoder = None  

    def fit(self, df):
        """
        Train the BERT model.
        Splits the data into X (features) and y (labels) internally.
        Returns accuracy, f1_score, true_labels, predictions
        """
        # Ensure the required columns exist
        if 'description' not in df.columns or 'industry' not in df.columns:
            raise ValueError("The input DataFrame must contain 'description' and 'industry' columns.")

        # Create label encoding
        unique_industries = df['industry'].unique()
        self.label_encoder = {label: idx for idx, label in enumerate(unique_industries)}
        self.inverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}

        # Extract features (X) and labels (y)
        print("Extracting features and labels...")
        X = df['description'].tolist()
        y = [self.label_encoder[ind] for ind in df['industry']]  # Convert industries to numeric labels

        # Split the data into training and testing sets
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Tokenize the training and testing data
        print("Tokenizing data...")
        train_encodings = self.tokenizer(
            X_train,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        test_encodings = self.tokenizer(
            X_test,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Convert labels to tensors
        train_labels = torch.tensor(y_train)
        test_labels = torch.tensor(y_test)

        # Create DataLoader for training and testing
        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
        test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        # Define optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        print("Training the BERT model...")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(train_loader):.4f}")

        # Evaluation loop
        print("Evaluating the model...")
        self.model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Convert numeric labels back to industry names
        true_labels_str = [self.inverse_label_encoder[label] for label in true_labels]
        predictions_str = [self.inverse_label_encoder[pred] for pred in predictions]

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        macro_f1 = f1_score(true_labels, predictions, average="macro")
        
        return accuracy, macro_f1, true_labels_str, predictions_str

    def predict(self, X):
        """Make predictions on new data."""
        self.model.eval()
        encodings = self.tokenizer(
            X,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

        # Convert numeric predictions to industry names
        if self.inverse_label_encoder:
            predictions = [self.inverse_label_encoder[pred] for pred in predictions]
            
        return predictions