import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.app.core.ner.bert_ner import BertNER
from src.app.core.classifiers.tf_idf import TFIDFWrapper
from src.app.api.routers import classify, extract_skills, health
from src.app.config.config import NER_MODEL_PATH, TFIDF_MODEL_PATH
from src.app.core.tokenizers.tokenizer_tf_idf import tokenizer_func

# Global variables to hold models
ner_model: BertNER | None = None
tfidf_model: TFIDFWrapper | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan for initializing models on startup
    and cleaning up on shutdown.
    """
    global ner_model, tfidf_model

    # --- Startup ---
    print("Loading models...")

    # Load NER model
    ner_model = BertNER()
    try:
        print(f"NER model initialized from {NER_MODEL_PATH}")
    except Exception:
        print("NER model not found, using default untrained model")

    # Load TF-IDF model
    try:
        tfidf_model = TFIDFWrapper.load(TFIDF_MODEL_PATH)
        print(f"TF-IDF model loaded from {TFIDF_MODEL_PATH}")
    except Exception:
        print("TF-IDF model not found, using default untrained model")
        tfidf_model = TFIDFWrapper()

    # Attach models to app.state for dependency injection
    app.state.ner_model = ner_model
    app.state.tfidf_model = tfidf_model

    yield  # App runs here

    # --- Shutdown ---
    print("Cleaning up models...")
    ner_model = None
    tfidf_model = None
    app.state.ner_model = None
    app.state.tfidf_model = None


# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan, title="Job Helper API", version="0.1.0")

# Include routers
app.include_router(classify.router)
app.include_router(extract_skills.router)
app.include_router(health.router)
