from fastapi import APIRouter, Depends
from src.app.api.dependencies.client import get_ner_model
from src.app.api.schemas.schemas import NERRequest
from src.app.api.responses.extract_skills import NERResponse

router = APIRouter()

@router.post("/extract-skills", response_model=NERResponse)
def extract_skills(request: NERRequest, model=Depends(get_ner_model)):
    """
    Endpoint to extract entities/skills from input texts using the NER model.
    """
    predictions = model.predict(request.texts)
    return NERResponse(entities=predictions)