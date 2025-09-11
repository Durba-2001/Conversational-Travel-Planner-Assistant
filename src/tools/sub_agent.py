# src/tools/sub_agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from src.agent.structure import TravelItinerary
from src.config import api_key
from src.tools.base import create_llm_chain


@tool
def generate_itinerary(user_request: str) -> TravelItinerary:
    """
    LLM-driven tool for generating a structured TravelItinerary.
    Accepts a user request text and returns a TravelItinerary object.
    No external tools are actually invoked — the LLM directly generates output.
    """
    # Prompt template with strict instructions
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful travel itinerary planner.
Always output **valid JSON only** that matches the TravelItinerary schema.
Rules:

- Always select a realistic destination based on the request.
- Ensure 'duration' matches the request.
- 'total_budget' must be a reasonable estimate (e.g. duration * 500 + 300).
- Daily activities should align with the interest mentioned in the request.
"""
    ),
    ("user", "{user_request}")
])


    # Create structured chain
    chain = create_llm_chain(prompt, structured=True, schema=TravelItinerary)

    # Run the chain
    itinerary: TravelItinerary = chain.invoke({"user_request": user_request})

    return itinerary.model_dump_json(indent=2)
