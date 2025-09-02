from pydantic import BaseModel, Field
from typing import List, Optional

class User(BaseModel):
    username: str = Field(..., description="Unique username")
    password_hash: str = Field(..., description="Hashed password")

class ChatInput(BaseModel):
    preference: str
    budget: float
    days: int
    interest: str

class Message(BaseModel):
    role: str = Field(..., description="Role of the sender (user/assistant/system)")
    content: str = Field(..., description="Message text")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp")

class Session(BaseModel):
    session_id: str = Field(..., description="Session UUID")
    username: str = Field(..., description="Owner username")
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")
    summary: Optional[str] = Field(None, description="Summarized old messages")
