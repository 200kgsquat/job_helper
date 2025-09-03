from pydantic import BaseModel
from typing import List, Dict

class NERResponse(BaseModel):
    entities: List[List[Dict]]
