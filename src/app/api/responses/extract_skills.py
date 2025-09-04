from fastapi import HTTPException, status
from typing import List
from src.app.api.schemas.schemas import NERResponse

extract_skills_responses = {
    200: {
        "description": "Success - skills are extracted successfully",
        "content": {
            "application/json": {
                "example": {
                    "entities": [
                        [
                            {"text": "Python", "type": "SKILL", "start": 0, "end": 6},
                            {"text": "Machine Learning", "type": "SKILL", "start": 10, "end": 26}
                        ],
                        [
                            {"text": "Data Analysis", "type": "SKILL", "start": 0, "end": 14},
                            {"text": "Deep Learning", "type": "SKILL", "start": 18, "end": 32}
                        ]
                    ]
                }
            }
        }
    },

    422: {
        "description": "Validation Error",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "texts"],
                            "msg": "field required",
                            "type": "value_error.missing"
                        }
                    ]
                }
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Internal server error during skill extraction"
                }
            }
        }
    }
}
