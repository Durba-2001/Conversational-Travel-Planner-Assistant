from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel
import json
from langchain.chains.base import Chain

# Load env
load_dotenv(find_dotenv())
api_key = os.environ.get("GOOGLE_API_KEY")

# Default shared LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)

# Optional default structured response
class LLMStructuredResponse(BaseModel):
    answer: str
    reason: str

def create_llm_chain(
    prompt_template: ChatPromptTemplate,
    structured: bool = False,
    schema: type[BaseModel] | None = None,
) -> Chain:
    """
    Create an LLM chain.
    If structured=True and schema is provided, output will conform to the schema.
    """
    if structured:
        if schema is None:
            schema = LLMStructuredResponse
        structured_llm = llm.with_structured_output(schema)
        return prompt_template | structured_llm
    return prompt_template | llm

def safe_parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
