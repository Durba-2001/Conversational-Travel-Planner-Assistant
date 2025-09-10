# src/tools/sub_agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from src.agent.structure import TravelItinerary
from src.config import api_key


@tool
def generate_itinerary(user_request: str) -> TravelItinerary:
    """
    LLM-driven tool for generating a structured TravelItinerary.
    Accepts a user request text and returns a TravelItinerary object.
    No external tools are actually invoked — the LLM directly generates output.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0
    )

    # Prompt template with strict instructions
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful travel itinerary planner.
Always output **valid JSON only** that matches the TravelItinerary schema:

{{
  "destination": "string",
  "duration": int,
  "total_budget": float,
  "daily_plans": [
    {{
      "duration": int,
      "activities": ["string", "string"],
      "estimated_cost": float
    }}
  ]
}}

Rules:
- Never output 'Unknown'.
- Always select a realistic destination based on the request.
- Ensure 'duration' matches the request.
- 'total_budget' must be a reasonable estimate (e.g. duration * 500 + 300).
- Daily activities should align with the interest mentioned in the request.
"""
    ),
    ("user", "{user_request}")
])


    # Create structured chain
    chain = prompt | llm.with_structured_output(schema=TravelItinerary)

    # Run the chain
    itinerary: TravelItinerary = chain.invoke({"user_request": user_request})

    return itinerary
