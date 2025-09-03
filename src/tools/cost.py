from pydantic import BaseModel, Field, field_validator
from langchain.prompts import ChatPromptTemplate
from src.tools.base import create_llm_chain
from langchain_core.tools import tool
import json

class CostEstimate(BaseModel):
    destination: str = Field(description="Selected destination")
    days: int = Field(description="Number of days")
    total_cost: float = Field(description="Estimated total cost in USD")
    breakdown: dict = Field(description="Cost components")

    @field_validator('breakdown', mode='before')
    def parse_breakdown(cls, v):
        """
        Parses the breakdown from a string into a dictionary.
        Handles both valid JSON and a custom 'key: value, ...' format.
        """
        try:
            # First, try to parse as valid JSON
            if v:
                return json.loads(v)
            else:
                raise ValueError("Breakdown must be a valid JSON dictionary string, but an empty string was provided.")
        except (json.JSONDecodeError, TypeError):
            # If JSON parsing fails, try to handle the custom format without regex
            try:
                parsed_dict = {}
                pairs = v.split(',')
                for pair in pairs:
                    key_value = pair.split(':', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip()
                        parsed_dict[key] = float(value)
                return parsed_dict
            except (ValueError, IndexError):
                raise ValueError("Breakdown must be a valid JSON dictionary string or a simple 'key: value' string.")

# Prompt template tells LLM what to produce
cost_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that estimates travel costs."),
    ("user", "Estimate travel cost for {days} days at {destination}. "
             "Return a JSON with total_cost (float), breakdown (dict of hotel, flight, other costs).")
])

# Create LLM chain for structured output tied to CostEstimate model
chain = create_llm_chain(cost_prompt, structured=True, schema=CostEstimate)

@tool
def estimate_cost(destination: str, days: int = None) -> str:
    """
    Estimate the travel cost for a given destination and number of days.

    Args:
        destination (str): The travel destination (or a JSON string containing destination and days).
        days (int): Number of days to stay.

    Returns:
        str: JSON string containing destination, days, total cost, and cost breakdown.
    """
    try:
        data = json.loads(destination)
        destination = data.get('destination', destination)
        days = data.get('days', days)
    except (json.JSONDecodeError, TypeError):
        pass

    if days is None:
        return json.dumps({
            "destination": destination,
            "days": None,
            "total_cost": 0.0,
            "breakdown": {"error": "Missing 'days' parameter. Please provide the number of days for the trip."}
        })
    
    try:
        days = int(days)
        if days <= 0:
            raise ValueError("Days must be a positive integer.")
    except (ValueError, TypeError):
        raise ValueError("Days must be a positive integer.")

    try:
        result: CostEstimate = chain.invoke({"destination": destination, "days": days})
    except Exception as e:
        print(f"LLM chain failed to produce a valid response: {e}")
        result = CostEstimate(
            destination=destination,
            days=days,
            total_cost=0.0,
            breakdown={"error": "LLM failed to generate valid cost data. Default values provided."}
        )
    return result.model_dump_json()
