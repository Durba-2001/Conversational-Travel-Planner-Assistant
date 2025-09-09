# tests/integration/test_chat.py
import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from types import SimpleNamespace
from unittest.mock import patch

from src.api.router import router
from src.database.session import get_db
from src.auth.router import get_current_user

# -----------------------
# Fake DB and user
# -----------------------
class FakeCollection:
    def __init__(self):
        self.data = []

    async def insert_one(self, doc):
        self.data.append(doc)
        return SimpleNamespace(inserted_id="123")

    async def find_one(self, query):
        for doc in self.data:
            if doc["session_id"] == query.get("session_id") and doc["username"] == query.get("username"):
                return doc
        return None

    async def update_one(self, query, update):
        doc = await self.find_one(query)
        if doc:
            doc["messages"].extend(update["$push"]["messages"]["$each"])
        return SimpleNamespace(modified_count=1)

fake_db = {"sessions": FakeCollection()}

async def override_get_db():
    return fake_db

async def override_current_user():
    return SimpleNamespace(username="testuser")

# -----------------------
# Setup test app
# -----------------------
app = FastAPI()
app.include_router(router)
app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = override_current_user
client = TestClient(app)

# -----------------------
# Patch run_agent to return fixed response
# -----------------------
@pytest.fixture(autouse=True)
def patch_run_agent():
    with patch("src.api.router.run_agent", return_value="Hello from assistant"):
        yield

# -----------------------
# Tests
# -----------------------
def test_create_chat():
    resp = client.post("/chat/", json={"message": "Hello"})
    assert resp.status_code == status.HTTP_201_CREATED
    data = resp.json()
    assert "session_id" in data
    assert data["response"] == "Hello from assistant"

def test_continue_chat():
    # Create a session first
    session_id = client.post("/chat/", json={"message": "Hi"}).json()["session_id"]

    # Continue chat
    resp = client.post(f"/chat/{session_id}", json={"message": "How are you?"})
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["session_id"] == session_id
    assert data["response"] == "Hello from assistant"

def test_continue_invalid_session():
    resp = client.post("/chat/invalid", json={"message": "Hi"})
    assert resp.status_code == status.HTTP_404_NOT_FOUND


def test_create_chat_empty_message():
    resp = client.post("/chat/", json={"message": ""})
    assert resp.status_code == status.HTTP_201_CREATED
    data = resp.json()
    assert data["response"] == "Hello from assistant"

def test_create_multiple_chats():
    session_ids = []
    for msg in ["Hi", "Hello", "How are you?"]:
        session_ids.append(client.post("/chat/", json={"message": msg}).json()["session_id"])
    assert len(set(session_ids)) == 3  # All session IDs are unique

def test_continue_chat_multiple_messages():
    session_id = client.post("/chat/", json={"message": "Hi"}).json()["session_id"]
    for msg in ["First", "Second", "Third"]:
        resp = client.post(f"/chat/{session_id}", json={"message": msg})
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["session_id"] == session_id
