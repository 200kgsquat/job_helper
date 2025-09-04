from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Classify schemas ---
class ClassifyRequest(BaseModel):
    texts: list[str]
    
class ClassifyResponse(BaseModel):
    predictions: List[str]

# --- NER / Extract skills schemas ---
class NERRequest(BaseModel):
    texts: List[str]

class Entity(BaseModel):
    entity_group: str = Field(..., description="Entity type (e.g., 'MISC', 'ORG')")
    score: float = Field(..., description="Model confidence score")
    word: str = Field(..., description="Extracted word/phrase")
    start: int = Field(..., description="Start position in text")
    end: int = Field(..., description="End position in text")
    index: Optional[int] = Field(None, description="Token index (optional)")

class NERResponse(BaseModel):
    entities: List[List[Entity]]