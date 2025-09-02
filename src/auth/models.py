from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UserSchema(BaseModel):
    username: str
    hashed_password: Optional[str] = None
    created_at: Optional[datetime] = None


class UserCreate(BaseModel):
    username: str
    password: str   # plain text password only for registration


class UserLogin(BaseModel):
    username: str
    password: str   # plain text password only for login

class UserResponse(BaseModel):
    username:str
    created_at: datetime
