
from src.tools.base import create_llm_chain, safe_parse_json
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from langchain_core.tools import tool


class DestinationSuggestion(BaseModel):
    destinations: List[str] = Field(
        description="List of recommended destinations"
    )
    reasoning: str = Field(
        description="Explanation for suggestions"
    )


# Prompt for the LLM
destination_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant."),
    ("user", "Suggest 3 destinations for a {preference} vacation within ${budget}. "
             "Return the answer strictly in JSON with keys 'destinations' and 'reasoning'.")
])

# Build chain from prompt
chain = create_llm_chain(destination_prompt)


@tool
def recommend_destinations(preference: str, budget: float) -> DestinationSuggestion:
    """
    Suggests travel destinations based on user preference and budget.

    Args:
        preference (str): The type of vacation, e.g., beach, adventure.
        budget (float): Available budget in USD.

    Returns:
        DestinationSuggestion: Recommended destinations and reasoning.
    """
    # Invoke the chain and safely parse the JSON output
    raw_output = chain.invoke({"preference": preference, "budget": budget})
    parsed = safe_parse_json(raw_output.content)
    
    # If parsing fails, return a default/empty object to prevent errors
    if parsed is None:
        return DestinationSuggestion(destinations=[], reasoning="Could not generate valid suggestions at this time.")

    return DestinationSuggestion(
        destinations=parsed.get("destinations", []),
        reasoning=parsed.get("reasoning", "")
    )
