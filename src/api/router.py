from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from src.agent.core import run_agent, run_stream_agent
from src.database.session import get_db
from src.auth.router import get_current_user
import uuid
from src.database.models import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["Chat"])


# --- Create new streaming session ---
@router.post("/stream")
async def create_stream_chat(
    request: ChatRequest,
    fastapi_request: Request,
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    session_id = str(uuid.uuid4())
    response_buffer = []

    # --- Save initial user message ---
    session_doc = {
        "session_id": session_id,
        "username": current_user.username,
        "messages": [{"role": "user", "message": request.message}],
    }
    result = await db["sessions"].insert_one(session_doc)
    if not result.inserted_id:
        raise HTTPException(status_code=500, detail="Failed to create session")

    async def event_generator():
        try:
            yield f"\n[Session ID]: {session_id}\n\n"
            async for chunk in run_stream_agent(request.message, session_id):
                    response_buffer.append(chunk)
                    yield f"{chunk}"
        except Exception as e:
            yield f"{str(e)}"

    # --- Swagger/Postman fallback → return JSON instead of streaming ---
    accept_header = fastapi_request.headers.get("accept", "").lower()
    user_agent = fastapi_request.headers.get("user-agent", "").lower()
    if "application/json" in accept_header or "swagger" in user_agent:
        async for _ in event_generator():
            pass
        final_response = " ".join(response_buffer).replace("\n", " ").strip()

        await db["sessions"].update_one(
            {"session_id": session_id},
            {"$push": {"messages": {"role": "assistant", "message": final_response}}}
        )
        return ChatResponse(session_id=session_id, response=final_response)

    # --- Streaming mode ---
    async def streaming_response():
        async for item in event_generator():
            yield item
        if response_buffer:
            final_response = " ".join(response_buffer).replace("\n", " ").strip()
            await db["sessions"].update_one(
                {"session_id": session_id},
                {"$push": {"messages": {"role": "assistant", "message": final_response}}}
            )

    return StreamingResponse(streaming_response(), media_type="text/event-stream")


# --- Continue existing session with streaming ---
@router.post("/stream/{session_id}")
async def continue_stream_chat(
    session_id: str,
    request: ChatRequest,
    fastapi_request: Request,
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    # Fetch session
    session = await db["sessions"].find_one({
        "session_id": session_id,
        "username": current_user.username
    })
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build context
    previous_messages = session.get("messages", [])
    context_text = "".join(f"{msg['role']}: {msg['message']}\n" for msg in previous_messages)
    full_input = context_text + f"user: {request.message}"

    response_buffer = []

    # Save user message
    await db["sessions"].update_one(
        {"session_id": session_id},
        {"$push": {"messages": {"role": "user", "message": request.message}}}
    )

    async def event_generator():
        try:
            async for chunk in run_stream_agent(full_input, session_id):
                    response_buffer.append(chunk)
                    yield f"{chunk}"
        except Exception as e:
            yield f"{str(e)}"

    # --- Swagger/Postman fallback → return JSON instead of streaming ---
    accept_header = fastapi_request.headers.get("accept", "").lower()
    user_agent = fastapi_request.headers.get("user-agent", "").lower()
    if "application/json" in accept_header or "swagger" in user_agent:
        async for _ in event_generator():
            pass
        final_response = " ".join(response_buffer).replace("\n", " ").strip()

        await db["sessions"].update_one(
            {"session_id": session_id},
            {"$push": {"messages": {"role": "assistant", "message": final_response}}}
        )
        return ChatResponse(session_id=session_id, response=final_response)

    # --- Streaming mode ---
    async def streaming_response():
        async for item in event_generator():
            yield item
        if response_buffer:
            final_response = " ".join(response_buffer).replace("\n", " ").strip()
            await db["sessions"].update_one(
                {"session_id": session_id},
                {"$push": {"messages": {"role": "assistant", "message": final_response}}}
            )

    return StreamingResponse(streaming_response(), media_type="text/event-stream")


# --- Create a new session (non-streaming) ---
@router.post("/", response_model=ChatResponse, status_code=201)
async def create_chat(
    request: ChatRequest,
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    session_id = str(uuid.uuid4())
    try:
        response_text = run_agent(request.message, session_id)  # synchronous
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


# --- Continue an existing session (non-streaming) ---
@router.post("/{session_id}", response_model=ChatResponse, status_code=200)
async def continue_chat(
    session_id: str,
    request: ChatRequest,
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

    try:
        response_text = run_agent(full_input, session_id)  # synchronous
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Agent error: {e}")

    await db["sessions"].update_one(
        {"session_id": session_id},
        {"$push": {"messages": {"$each": [
            {"role": "user", "message": request.message},
            {"role": "assistant", "message": response_text}
        ]}}}
    )

    return ChatResponse(session_id=session_id, response=response_text)
