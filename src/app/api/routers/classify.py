from fastapi import APIRouter, Depends
from src.app.api.dependencies.client import get_tfidf_model
from src.app.api.schemas.schemas import ClassifyRequest
from src.app.api.responses.classify import ClassifyResponse

router = APIRouter()

@router.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest, model=Depends(get_tfidf_model)):
    """
    Endpoint to classify input texts using the TF-IDF model.
    """
    predictions = model.predict(request.texts)
    return ClassifyResponse(predictions=predictions)

