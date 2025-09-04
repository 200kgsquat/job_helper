from fastapi import HTTPException, status
from typing import List
from src.app.api.schemas.schemas import ClassifyResponse

classify_responses = {
    200: {
        "description": "Success - texts are classified successfully",
        "content": {
            "application/json": {
                "example": {
                    "predictions": 
                    [
                    "IT"
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
                    "detail": "Internal server error during classification"
                }
            }
        }
    }
}

