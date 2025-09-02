import pytest
import pandas as pd
import numbers
from src.app.core.classifiers.tf_idf import TFIDFWrapper

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "description": [
            "python developer with machine learning",
            "school teacher and education",
            "marketing specialist in SEO"
        ],
        "industry": [
            "Software",
            "Education",
            "Marketing"
        ]
    })

def test_tfidfwrapper_fit_and_predict(sample_df):
    wrapper = TFIDFWrapper(max_features=50, ngram_range=(1, 1))
    wrapper.fit(sample_df)

    X_test = [
        "machine learning engineer",
        "high school teacher",
        "SEO and marketing expert"
    ]

    predictions = wrapper.predict(X_test)

    assert len(predictions) == len(X_test)
    assert all(isinstance(p, numbers.Number) for p in predictions)

def test_save_and_load(sample_df, tmp_path):
    wrapper = TFIDFWrapper(max_features=50, ngram_range=(1,1))
    wrapper.fit(sample_df)

    save_path = tmp_path / "tfidf_pipeline.joblib"
    wrapper.save(save_path)

    loaded_wrapper = TFIDFWrapper.load(save_path)
    X_test = ["machine learning engineer"]
    predictions = loaded_wrapper.predict(X_test)

    assert len(predictions) == 1
    assert isinstance(predictions[0], numbers.Number)
