from pydantic import BaseModel, Field


class User(BaseModel):
    username: str = Field(..., description="Unique username")
    password_hash: str = Field(..., description="Hashed password")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str  # Human-readable text

