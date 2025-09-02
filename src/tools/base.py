from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv,find_dotenv
from langchain.prompts import ChatPromptTemplate
import json
from langchain.chains.base import Chain
load_dotenv(find_dotenv())
api_key = os.environ.get("GOOGLE_API_KEY")




# Shared LLM instance for all tools
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=api_key)

def create_llm_chain(prompt_template: ChatPromptTemplate) -> Chain:
    
    return prompt_template | llm

def safe_parse_json(text: str):
   
    try:
        return json.loads(text)        # Safely parse JSON from LLM output text, returning None on failure.

    except json.JSONDecodeError:
        return None
