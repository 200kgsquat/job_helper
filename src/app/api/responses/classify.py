from pydantic import BaseModel
from typing import List

class ClassifyResponse(BaseModel):
    predictions: List[str]
