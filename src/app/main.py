from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from moc.BEER import BEER
# from moc.NEGR import NEGR

app = FastAPI(title="Text Classification & Skill Extraction API")

class TextRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    category: str

class SkillsResponse(BaseModel):
    skills: list[str]

@app.post("/classify", response_model=ClassificationResponse)
def classify_endpoint(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return ClassificationResponse(category='Data Science')

@app.post("/extract-skills", response_model=SkillsResponse)
def extract_skills_endpoint(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return SkillsResponse(skills=['Python', 'Machine Learning'])


