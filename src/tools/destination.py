from src.tools.base import create_llm_chain, safe_parse_json
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from langchain_core.tools import tool
import json

class DestinationSuggestion(BaseModel):
    destinations: List[str] = Field(
        description="List of recommended destinations"
    )
    reasoning: str = Field(
        description="Explanation for suggestions"
    )


destination_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. You suggest travel destinations based on user preferences and budget. Return the output in a JSON structure conforming to the DestinationSuggestion schema."),
    ("user", "Suggest 3 destinations for a {preference} vacation within a ${budget} budget."),
])

chain = create_llm_chain(destination_prompt, structured=True, schema=DestinationSuggestion)

@tool
def recommend_destinations(preference: str, budget: float = None) -> str:
    """
    Suggests travel destinations based on user preference and budget.

    Args:
        preference (str): The type of vacation, e.g., 'beach', 'adventure', 'relaxation'.
        budget (float): Available budget in USD.

    Returns:
        str: A JSON string of recommended destinations and reasoning.
    """
    # Check if the preference input is a JSON string containing both values
    try:
        data = json.loads(preference)
        if 'preference' in data and 'budget' in data:
            preference = data['preference']
            budget = data['budget']
    except json.JSONDecodeError:
        pass  # It's not a JSON string, so we continue with the original preference and budget values

    # Check if the budget is valid before invoking the LLM chain
    # if not isinstance(budget, (int, float)) or budget <= 0:
    #     raise ValueError("Budget must be a positive number.")

    result_obj: DestinationSuggestion = chain.invoke({"preference": preference, "budget": budget})
    return result_obj.model_dump_json(indent=2)
