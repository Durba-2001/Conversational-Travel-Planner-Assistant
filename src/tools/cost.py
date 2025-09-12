from langchain_core.tools import tool
import json
from src.agent.structure import CostEstimate

@tool
def estimate_cost(inputs: dict | str) -> str:
    """
    Estimates total travel cost in USD.
    Formula: total_cost = (days * 500) + 300
    (hotel: $500 per day + base flight: $300).

    Args:
        inputs (dict | str): Dictionary with keys:
            - 'destination' (str): The travel destination
            - 'days' (int): Number of days for the trip(take number of days as 1 if nothing is mentioned.)

    Returns:
        str: JSON string matching CostEstimate schema,
             including breakdown of hotel and flight.
    """
    # If input is a string, parse it
    if isinstance(inputs, str):
        inputs = json.loads(inputs)

    # Handle nested {"inputs": {...}}
    if "inputs" in inputs:
        inputs = inputs["inputs"]

    destination = inputs.get("destination", "Unknown Destination")
    days = int(inputs.get("days", 1))  # default to 1 day if missing

    # Simple static cost calculation
    hotel_cost = 500 * days
    flight_cost = 300
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
