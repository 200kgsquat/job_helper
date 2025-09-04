import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.app.core.ner.bert_ner import BertNER
from src.app.core.classifiers.tf_idf import TFIDFWrapper
from src.app.api.routers import classify, extract_skills, health
from src.app.config.config import NER_MODEL_PATH, TFIDF_MODEL_PATH
from src.app.logs.logger import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan for initializing models on startup
    and cleaning up on shutdown.
    """
    # --- Startup ---
    logger.info("Loading models...")

    # Load NER model
    try:
        ner_model = BertNER()
        logger.info(f"NER model initialized from {NER_MODEL_PATH}")
    except Exception as e:
        logger.warning(f"NER model not found, using default untrained model: {e}")

    # Load TF-IDF model
    try:
        tfidf_model = TFIDFWrapper.load(TFIDF_MODEL_PATH)
        logger.info(f"TF-IDF model loaded from {TFIDF_MODEL_PATH}")
    except Exception as e:
        logger.warning(f"TF-IDF model not found, using default untrained model: {e}")
        tfidf_model = TFIDFWrapper()

    # Attach models to app.state for dependency injection
    app.state.ner_model = ner_model
    app.state.tfidf_model = tfidf_model

    yield  # App runs here

    # --- Shutdown ---
    logger.info("Cleaning up models...")
    # Python's garbage collection will handle cleanup automatically
    # No need to manually set variables to None


# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan, title="Job Helper API", version="0.1.0")

# Include routers
app.include_router(classify.router)
app.include_router(extract_skills.router)
app.include_router(health.router)