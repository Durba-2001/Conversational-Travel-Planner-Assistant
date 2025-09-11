from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel
import json
from langchain.chains.base import Chain
from src.config import api_key

# Default shared LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=api_key)

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
