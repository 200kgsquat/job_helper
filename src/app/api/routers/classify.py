from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.app.api.dependencies.client import get_bert_model

router = APIRouter()

class ClassifyRequest(BaseModel):
    texts: list[str]

class ClassifyResponse(BaseModel):
    predictions: list[str]

@router.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest, model=Depends(get_bert_model)):
    predictions = model.predict(request.texts)
    return ClassifyResponse(predictions=predictions)
