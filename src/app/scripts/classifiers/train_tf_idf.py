# src/app/scripts/classifiers/train_tf_idf.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from src.app.config import CLEANED_DATA_FILE, TFIDF_MODEL_PATH

# adjust import path if needed
from src.app.core.tokenizers.tokenizer_tf_idf import TextCleaner

# Initialize the text cleaner
cleaner = TextCleaner()

# Top-level tokenizer function (needed for pickling)
def tokenizer_func(text):
    cleaned = cleaner.clean_text(text)
    return cleaned.split() if cleaned else []

def main():
    cleaned_data_file = CLEANED_DATA_FILE
    model_save_path = TFIDF_MODEL_PATH

    print("Loading cleaned data...")
    df = pd.read_csv(data_path)

    # Ensure descriptions are strings
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df = df[df["description"].str.len() > 0].reset_index(drop=True)
    df["industry"] = df["industry"].fillna("unknown").astype(str)

    X = df["description"].tolist()
    y = df["industry"].tolist()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("Fitting TF-IDF + Logistic Regression pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            tokenizer=tokenizer_func,
            preprocessor=None,
            token_pattern=None
        )),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    print("Predicting on test set...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nEvaluation on test set:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}\n")

    print("Classification report (test set):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save pipeline
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\nPipeline saved to {model_path}")

    # Demo predictions
    print("\nDemo predictions:")
    examples = [
        "Senior Python developer with experience in Django and AWS",
        "Looking for a school teacher with curriculum development experience",
        "Marketing manager with experience in digital advertising and SEO"
    ]
    preds = pipeline.predict(examples)
    for txt, p in zip(examples, preds):
        print(f"Text: {txt}\nPredicted industry: {p}\n")


if __name__ == "__main__":
    main()
