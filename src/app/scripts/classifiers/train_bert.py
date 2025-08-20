import os
import pandas as pd
from sklearn.metrics import classification_report
from src.app.core.classifiers.bert import BertWrapper
import torch
from src.app.config import CLEANED_DATA_FILE, BERT_MODEL_PATH

def main():
    # Define file paths
    cleaned_data_file = CLEANED_DATA_FILE
    model_save_path = BERT_MODEL_PATH

    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv(cleaned_data_file)

    # Initialize BERT wrapper
    print("Initializing BERT model...")
    num_labels = len(df["industry"].unique())
    bert_model = BertWrapper(
        model_name='bert-base-uncased',
        num_labels=num_labels,
        epochs=3,
        batch_size=16,
        learning_rate=5e-5
    )

    # Train the model and get evaluation results
    print("Training BERT model...")
    accuracy, macro_f1, true_labels, predictions = bert_model.fit(df)

    # Print evaluation results
    print("\nEvaluation on test set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}\n")

    print("Classification report (test set):")
    print(classification_report(true_labels, predictions, zero_division=0))

    # Demo predictions
    print("\nDemo predictions:")
    examples = [
        "Senior Python developer with experience in Django and AWS",
        "Looking for a school teacher with curriculum development experience",
        "Marketing manager with experience in digital advertising and SEO"
    ]
    preds = bert_model.predict(examples)
    for txt, p in zip(examples, preds):
        print(f"Text: {txt}\nPredicted industry: {p}\n")

    # Ensure the model folder exists
    os.makedirs("model", exist_ok=True)

    # Save the trained model
    print("Saving the trained model...")
    torch.save({
        'model_state_dict': bert_model.model.state_dict(),
        'label_encoder': bert_model.label_encoder,
        'inverse_label_encoder': bert_model.inverse_label_encoder
    }, model_save_path)
    print(f"Model saved to '{model_save_path}'.")

if __name__ == "__main__":
    main()