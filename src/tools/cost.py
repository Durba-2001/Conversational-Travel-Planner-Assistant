from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from src.tools.base import create_llm_chain,safe_parse_json
from langchain_core.tools import tool


class CostEstimate(BaseModel):
    destination: str = Field(description="Selected destination")
    days: int = Field(description="Number of days")
    total_cost: float = Field(description="Estimated total cost in USD")
    breakdown: dict = Field(description="Cost components")

cost_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that estimates travel costs."),
    ("user", "Estimate travel cost for {days} days at {destination}. Return a JSON with total_cost (float), breakdown (dict of hotel, flight, other costs).")
])
chain = create_llm_chain(cost_prompt)

@tool
def estimate_cost(destination: str, days: int)-> CostEstimate:
    """
    Estimate the travel cost for a given destination and number of days.
    
    Args:
        destination (str): The travel destination.
        days (int): Number of days to stay.

    Returns:
        CostEstimate: Contains the destination, days, total cost, and cost breakdown.
    """
    result = chain.invoke({"destination": destination, "days": days})
    parsed = safe_parse_json(result)
    total_cost = 0.0
    breakdown = {}
    if parsed:
        total_cost = parsed.get("total_cost", 0.0)
        breakdown = parsed.get("breakdown", {})
    return CostEstimate(destination=destination, days=days, total_cost=total_cost, breakdown=breakdown)