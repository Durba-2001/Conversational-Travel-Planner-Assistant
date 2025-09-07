from pydantic import BaseModel, Field
from langchain_core.tools import tool
import json

class CostEstimate(BaseModel):
    destination: str = Field(description="Selected destination")
    days: int = Field(description="Number of days")
    total_cost: float = Field(description="Estimated cost in USD")
    breakdown: dict = Field(description="Cost components")

@tool
def estimate_cost(inputs: "dict | str") -> str:
    """
    Estimates total travel cost based on a simple formula:
    total_cost = (days * 10000) + 20000 (hotel + base flight cost).

    Args:
        inputs (dict | str): Dictionary with keys:
            - 'destination' (str): The travel destination
            - 'days' (int): Number of days for the trip

    Returns:
        str: JSON string matching CostEstimate schema, including breakdown of hotel and flight.
    """
    # If input is string, parse it
    if isinstance(inputs, str):
        inputs = json.loads(inputs)

    destination = inputs.get("destination", "Unknown Destination")
    days = int(inputs.get("days", 1))  # default to 1 day if missing

    # Simple static cost calculation
    hotel_cost = 500 * days  # e.g., 500 USD per day
    flight_cost = 300  # base flight cost
    total_cost = hotel_cost + flight_cost

    breakdown = {
        "hotel": round(hotel_cost, 2),
        "flight": round(flight_cost, 2)
    }

    estimate = CostEstimate(
        destination=destination,
        days=days,
        total_cost=round(total_cost, 2),
        breakdown=breakdown
    )

    return estimate.model_dump_json(indent=2)
