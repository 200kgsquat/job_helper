from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    category: str

class SkillsResponse(BaseModel):
    skills: list[str]