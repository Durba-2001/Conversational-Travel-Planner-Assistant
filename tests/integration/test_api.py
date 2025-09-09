import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends
from src.api.router import router as chat_router
from src.auth.router import get_current_user
from src.database.session import get_db
from pydantic import BaseModel
from unittest.mock import AsyncMock
import uuid

# ---- Mock current_user ----
class MockUser:
    username = "testuser"

async def override_get_current_user():
    return MockUser()

# ---- Mock DB ----
class FakeDB:
    def __init__(self):
        self.sessions = []

    async def insert_one(self, doc):
        doc["inserted_id"] = str(uuid.uuid4())
        self.sessions.append(doc)
        return doc

    async def find_one(self, query):
        for s in self.sessions:
            if s["session_id"] == query.get("session_id") and s["username"] == query.get("username"):
                return s
        return None

    async def update_one(self, query, update):
        session = await self.find_one(query)
        if session:
            for msg in update["$push"]["messages"]["$each"]:
                session["messages"].append(msg)
        return session

async def override_get_db():
    return FakeDB()

# ---- Mock run_agent ----
def mock_run_agent(message, session_id):
    return f"Echo: {message}"

# ---- Setup FastAPI app ----
app = FastAPI()
app.include_router(chat_router, prefix="/chat")
app.dependency_overrides[get_current_user] = override_get_current_user
app.dependency_overrides[get_db] = override_get_db

# Patch the agent
import src.api.router as chat_module
chat_module.run_agent = mock_run_agent

client = TestClient(app)

# ---- TEST CASES ----
def test_create_chat():
    payload = {"message": "Hello AI!"}
    resp = client.post("/chat/", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["response"] == "Echo: Hello AI!"
    assert "session_id" in data

def test_continue_chat():
    # First, create a session
    payload = {"message": "Start chat"}
    create_resp = client.post("/chat/", json=payload)
    session_id = create_resp.json()["session_id"]

    # Continue the session
    continue_payload = {"message": "Continue chat"}
    resp = client.post(f"/chat/{session_id}", json=continue_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["response"] == "Echo: user: Start chat\nuser: Continue chat"
    assert data["session_id"] == session_id
