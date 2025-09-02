# src/app/core/classifiers/tf_idf.py
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.app.core.tokenizers.tokenizer_tf_idf import TextCleaner

text_cleaner = TextCleaner()

def tokenizer_func(text):
    return text_cleaner.clean_text(text)

class TFIDFWrapper:
    def __init__(self, max_features=10000, ngram_range=(1, 2), classifier=None, test_size=0.2, random_state=42):
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        # Initialize classifier (default: Logistic Regression)
        self.classifier = classifier if classifier else LogisticRegression(max_iter=1000)
        # Create a pipeline
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.classifier)
        ])
        # Parameters for train-test split
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, df):
        """
        Train the TF-IDF model and classifier.
        Splits the data into X (features) and y (labels) internally.
        """
        # Ensure the required columns exist
        if 'description' not in df.columns or 'industry' not in df.columns:
            raise ValueError("The input DataFrame must contain 'description' and 'industry' columns.")

        # Extract features (X) and labels (y)
        print("Extracting features and labels...")
        X = df['description'].tolist()
        y = df['industry'].astype('category').cat.codes.tolist()  # Convert industries to numeric labels

        # Split the data into training and testing sets
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Train the model on the training set
        print("Training the TF-IDF model...")
        self.pipeline.fit(X_train, y_train)

        # Evaluate the model on the test set
        print("Evaluating the model...")
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy on test set: {accuracy:.2f}")

    def predict(self, X):
        """Make predictions on new data."""
        return self.pipeline.predict(X)

    def save(self, path: str):
        """
        Save the TF-IDF pipeline to a .joblib file.
        """
        joblib.dump(self.pipeline, path)
        print(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str):
        """
        Load a pre-trained TF-IDF pipeline from a .joblib file.
        """
        pipeline = joblib.load(path)
        instance = cls()
        instance.pipeline = pipeline
        return instance
