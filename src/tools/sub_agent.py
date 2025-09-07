# src/tools/sub_agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.tools import tool
from src.agent.structure import TravelItinerary
import json
import os
from dotenv import load_dotenv
@tool
def generate_itinerary(user_request: str) -> TravelItinerary:
    """
    LLM-driven ReAct sub-agent for generating a structured TravelItinerary.
    Accepts a user request text, calls internal tools via LLM reasoning, 
    and returns structured TravelItinerary.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0)

    prompt = f"""
You are an itinerary generator agent.
Given the user request: "{user_request}", do the following:

1. Extract preferences, budget, duration, and interest.
2. Call the internal tools in order using their exact tool names:
   - Destination Recommender (preference, budget)
   - Cost Estimator (destination, duration)
   - Activity Planner (destination, interest, duration, budget)
3. Combine all results into a structured JSON:

{{
  "destination": "<destination>",
  "duration_days": <duration>,
  "total_budget": <total_budget>,
  "daily_plans": [
      {{"day": 1, "activities": ["..."], "estimated_cost": ...}},
      {{"day": 2, "activities": ["..."], "estimated_cost": ...}}
  ]
}}

Respond ONLY in valid JSON. Use the exact tool names above when referencing tools.
"""


    response = llm([HumanMessage(content=prompt)]).content

    # Safe JSON parsing
    try:
        itinerary_dict = json.loads(response)
    except json.JSONDecodeError:
        itinerary_dict = {
            "destination": "Unknown",
            "duration_days": 1,
            "total_budget": 0.0,
            "daily_plans": []
        }

    return TravelItinerary(
        destination=itinerary_dict.get("destination"),
        duration=itinerary_dict.get("duration_days"),
        total_budget=itinerary_dict.get("total_budget"),
        daily_plans=itinerary_dict.get("daily_plans", []),
    )
