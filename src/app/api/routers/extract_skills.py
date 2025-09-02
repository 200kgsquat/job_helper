from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.app.api.dependencies.client import get_ner_model

router = APIRouter()

class NERRequest(BaseModel):
    texts: list[str]

class NERResponse(BaseModel):
    entities: list[list[dict]]

@router.post("/extract-skills", response_model=NERResponse)
def extract_skills(request: NERRequest, model=Depends(get_ner_model)):
    predictions = model.predict(request.texts)
    return NERResponse(entities=predictions)
