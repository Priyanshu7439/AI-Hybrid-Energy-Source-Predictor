from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="AI Hybrid Energy API"
)

app.include_router(router)