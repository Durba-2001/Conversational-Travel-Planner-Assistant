from fastapi import FastAPI
from src.api.router import router as api_router
from src.auth.router import router as auth_router

app = FastAPI(title="Travel Planner Assistant",version="1.0",description="conversation travel assistant")

app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(api_router, prefix="/api", tags=["Chat"])
