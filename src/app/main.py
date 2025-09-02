from fastapi import FastAPI
from src.app.api.routers import classify, extract_skills, health

app = FastAPI()

app.include_router(classify.router)
app.include_router(extract_skills.router)
app.include_router(health.router)
