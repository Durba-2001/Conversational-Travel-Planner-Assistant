from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from src.agent.core import run_agent
from src.database.session import get_db
from src.auth.router import get_current_user
import uuid
from src.database.models import ChatRequest,ChatResponse
router = APIRouter(prefix="/chat", tags=["Chat"])

# Request / Response models
# class ChatRequest(BaseModel):
#     message: str

# class ChatResponse(BaseModel):
#     session_id: str
#     response: str  # Human-readable text

# Create a new session
@router.post("/", response_model=ChatResponse, status_code=201)
async def create_chat(request: ChatRequest, db=Depends(get_db), current_user=Depends(get_current_user)):
    session_id = str(uuid.uuid4())
    try:
        response_text = run_agent(request.message, session_id)  #  synchronous
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Agent error: {e}")

    session_doc = {
        "session_id": session_id,
        "username": current_user.username,
        "messages": [
            {"role": "user", "message": request.message},
            {"role": "assistant", "message": response_text},
        ],
    }
    result = await db["sessions"].insert_one(session_doc)
    if not result.inserted_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create session")

    return ChatResponse(session_id=session_id, response=response_text)

# Continue an existing session
@router.post("/{session_id}", response_model=ChatResponse, status_code=200)
async def continue_chat(session_id: str, request: ChatRequest, db=Depends(get_db), current_user=Depends(get_current_user)):
    # Fetch session
    session = await db["sessions"].find_one({"session_id": session_id, "username": current_user.username})
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    # --- Build full context for agent ---
    previous_messages = session.get("messages", [])
    context_text = ""
    for msg in previous_messages:
        context_text += f"{msg['role']}: {msg['message']}\n"

    # Include the new user message at the end
    full_input = context_text + f"user: {request.message}"

    try:
        # Run agent with full context (synchronous)
        response_text = run_agent(full_input, session_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Agent error: {e}")

    # Append messages to DB
    await db["sessions"].update_one(
        {"session_id": session_id},
        {"$push": {"messages": {"$each": [
            {"role": "user", "message": request.message},
            {"role": "assistant", "message": response_text}
        ]}}}
    )

    return ChatResponse(session_id=session_id, response=response_text)
