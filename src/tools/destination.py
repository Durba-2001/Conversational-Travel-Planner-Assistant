# src/tools/destination.py
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from src.tools.base import create_llm_chain
from langchain_core.tools import tool
from typing import List
import json
from src.agent.structure import DestinationSuggestion
# class DestinationSuggestion(BaseModel):
#     destinations: List[str] = Field(description="List of recommended destinations")
#     reasoning: str = Field(description="Explanation for suggestions")

destination_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Suggest destinations based on preferences and budget. Return JSON."),
    ("user", "Suggest 3 destinations for a {preference} trip within a ${budget} budget.")
])

chain = create_llm_chain(destination_prompt, structured=True, schema=DestinationSuggestion)

@tool
def recommend_destinations(preference: str, budget: float = None) -> str:
    """
    Suggest travel destinations based on user preferences and optional budget.

    Args:
        preference (str): User's travel interest, e.g., 'historical', 'beach', 'adventure'.
        budget (float, optional): Maximum budget for the trip.

    Returns:
        str: JSON string containing a list of recommended destinations and reasoning.
    """
    try:
        data = json.loads(preference)
        if 'preference' in data and 'budget' in data:
            preference = data['preference']
            budget = data['budget']
    except json.JSONDecodeError:
        pass

    result_obj: DestinationSuggestion = chain.invoke({"preference": preference, "budget": budget})
    return result_obj.model_dump_json(indent=2)
