# src/api/chat_router.py

from fastapi import APIRouter, Body, Depends, HTTPException, status
from src.agent.core import run_agent
from src.database.session import get_db
from src.auth.router import get_current_user
from pydantic import BaseModel
import uuid

router = APIRouter(prefix="/chat", tags=["Chat"])

#  Define models for request/response
class ChatMessage(BaseModel):
    preference: str
    budget: float
    days: int
    interest: str

class ChatResponse(BaseModel):
    session_id: str | None = None
    response: str

@router.post("/", response_model=ChatResponse, status_code=201)
async def create_chat(
    message: ChatMessage,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    session_id = str(uuid.uuid4())
    try:
        itinerary = run_agent(
            preference=message.preference,
            budget=message.budget,
            duration=message.days,
            interest=message.interest,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent error: {str(e)}"
        )

    new_session = {
        "session_id": session_id,
        "username": current_user.username,
        "messages": [message.dict()],
        "summary": None,
    }
    result = await db["sessions"].insert_one(new_session)
    if not result.inserted_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session"
        )

    return ChatResponse(session_id=session_id, response=itinerary)

@router.post("/{session_id}", response_model=ChatResponse, status_code=200)
async def continue_chat(
    session_id: str,
    message: ChatMessage,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    update_result = await db["sessions"].update_one(
        {"session_id": session_id, "username": current_user.username},
        {"$push": {"messages": message.dict()}},
    )
    if update_result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found."
        )

    try:
        itinerary = run_agent(
            preference=message.preference,
            budget=message.budget,
            duration=message.days,
            interest=message.interest,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent error: {str(e)}"
        )

    return ChatResponse(response=itinerary)
