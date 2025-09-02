from langchain.memory import ConversationSummaryMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize your LLM for summarization (should match your main LLM or could be a smaller/cheaper one for efficiency)
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
api_key = os.environ.get("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=api_key)
# Create a conversation memory with summarization
def create_memory(llm,max_messages: int = 10, summary_token_budget: int = 300):
    return ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=summary_token_budget
    )