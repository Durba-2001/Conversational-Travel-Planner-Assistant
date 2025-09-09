from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from src.config import api_key,MongoDB_url
# LLM for summarization (can be same or cheaper model)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=api_key)

# Synchronous session memories cache
session_memories = {}

def get_session_memory(session_id: str, summary_token_budget: int = 300):
    """
    Returns a session-specific ConversationSummaryBufferMemory,
    backed by MongoDB for persistence.
    """
    if session_id in session_memories:
        return session_memories[session_id]

    # MongoDBChatMessageHistory can be used synchronously here
    chat_history = MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MongoDB_url,
        database_name="travel_planner",  # your DB name
        collection_name="sessions",
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=chat_history,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=summary_token_budget,
    )

    session_memories[session_id] = memory
    return memory
