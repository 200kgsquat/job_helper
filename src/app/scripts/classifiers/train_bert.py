import os
import pandas as pd
from transformers import BertTokenizer
from src.app.core.classifiers.bert import BertWrapper
import torch

def main():
    # Define file paths
    cleaned_data_file = "data/cleaned_job_postings.csv"
    model_save_path = "model/bert_model.pth"

    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv(cleaned_data_file)

    # Extract features and labels
    descriptions = df['description'].tolist()
    industries = df['industry'].astype('category').cat.codes.tolist()  # Convert industries to numeric labels

    # Initialize BERT tokenizer
    print("Initializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the descriptions
    print("Tokenizing descriptions...")
    tokenized = tokenizer(
        descriptions,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Extract input IDs and attention masks
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']

    # Initialize BERT wrapper
    print("Initializing BERT model...")
    bert_model = BertWrapper(
        model_name='bert-base-uncased',
        num_labels=len(set(industries)),  # Number of unique industries
        epochs=3,
        batch_size=16,
        learning_rate=5e-5
    )

    # Train the model
    print("Training BERT model...")
    bert_model.fit(input_ids, attention_mask, industries)

    print("Training complete!")

    # Ensure the model folder exists
    os.makedirs("model", exist_ok=True)

    # Save the trained model
    print("Saving the trained model...")
    torch.save(bert_model.model.state_dict(), model_save_path)
    print(f"Model saved to '{model_save_path}'.")

if __name__ == "__main__":
    main()