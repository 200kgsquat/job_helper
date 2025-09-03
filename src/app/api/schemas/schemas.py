from pydantic import BaseModel

# --- Classify schemas ---
class ClassifyRequest(BaseModel):
    texts: list[str]
    
# --- NER / Extract skills schemas ---
class NERRequest(BaseModel):
    texts: list[str]
