from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from src.agent.core import run_agent, run_stream_agent
from src.database.session import get_db
from src.auth.router import get_current_user
import uuid
from src.database.models import ChatRequest,ChatResponse
router = APIRouter(prefix="/chat", tags=["Chat"])

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse, JSONResponse
from src.agent.core import run_agent, run_stream_agent
from src.database.session import get_db
from src.auth.router import get_current_user
import uuid
from src.database.models import ChatRequest, ChatResponse
import asyncio
router = APIRouter(prefix="/chat", tags=["Chat"])

# --- Stream new session ---
from fastapi import Request
@router.post("/stream")
async def create_stream_chat(
    request: ChatRequest,
    fastapi_request: Request,
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    
    session_id = str(uuid.uuid4())

    # --- Async event generator for CMD streaming ---
    async def event_generator():
        try:
            async for chunk in run_stream_agent(request.message, session_id):
               
                yield f"{chunk}"
                await asyncio.sleep(0)  # force flush each token
        except Exception as e:
            yield f"[Error] {e}"
            await asyncio.sleep(0)

    # --- Store session ---
    session_doc = {
        "session_id": session_id,
        "username": current_user.username,
        "messages": [{"role": "user", "message": request.message}],
    }
    result = await db["sessions"].insert_one(session_doc)
    if not result.inserted_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session"
        )

    # --- Swagger → final JSON, CMD → live stream ---
    user_agent = fastapi_request.headers.get("user-agent", "").lower()
    if "swagger" in user_agent or "mozilla" in user_agent:
        response_text = run_agent(request.message, session_id)
        return ChatResponse(session_id=session_id, response=response_text)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- Continue existing session with streaming ---
@router.post("/stream/{session_id}")
async def continue_stream_chat(
    session_id: str,
    request: ChatRequest,
    req: Request,
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    # Fetch session
    session = await db["sessions"].find_one({"session_id": session_id, "username": current_user.username})
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    # Build context
    previous_messages = session.get("messages", [])
    context_text = "".join(f"{msg['role']}: {msg['message']}\n" for msg in previous_messages)
    full_input = context_text + f"user: {request.message}"

    full_response = ""

    async def event_generator():
        nonlocal full_response
        try:
            async for chunk in run_stream_agent(full_input, session_id):  # ✅ async for
                if chunk.startswith("@@FINAL@@"):
                    full_response = chunk.replace("@@FINAL@@", "").strip()
                else:
                    yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: [Error] {e}\n\n"

    # Save user message
    await db["sessions"].update_one(
        {"session_id": session_id},
        {"$push": {"messages": {"role": "user", "message": request.message}}}
    )

    # Swagger fallback → return JSON
    if "swagger" in str(req.headers.get("user-agent", "")).lower():
        async for _ in event_generator():
            pass
        return ChatResponse(session_id=session_id, response=full_response)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
