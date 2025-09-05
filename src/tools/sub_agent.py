from langchain.tools import tool
from src.agent.structure import TravelItinerary
from src.tools.destination import recommend_destinations
from src.tools.cost import estimate_cost
from src.tools.activity import plan_activities
import json

@tool
def generate_itinerary(input_data) -> TravelItinerary:
    """Generate a detailed travel itinerary based on preferences, budget, days, and interest."""

    # Safely handle both dict and JSON string inputs
    if isinstance(input_data, str):
        try:
            input_data = json.loads(input_data)
        except Exception as e:
            raise ValueError(f"Input must be a dict or valid JSON string: {e}")

    preference = input_data.get("preference")
    budget = input_data.get("budget")
    duration = input_data.get("duration")
    interest = input_data.get("interest")

    # Explicit validation (don't use all([...]) for numeric fields)
    if preference is None or budget is None or duration is None or interest is None:
        raise ValueError(f"Missing one or more required fields: {input_data}")

    # Call helper tools directly
    dest_result = recommend_destinations({"preference": preference,"budget":budget})
    destination = dest_result.destinations[0] if dest_result.destinations else "Unknown"

    cost_result = estimate_cost({"destination": destination, "duration": duration})
    total_budget = cost_result.budget

    activities_result = plan_activities({
        "destination": destination,
        "interest": interest,
        "duration": duration
    })
    daily_plans = activities_result.daily_plans

    # Build the itinerary model
    return TravelItinerary(
        destination=destination,
        duration=duration,
        total_budget=total_budget,
        daily_plans=daily_plans,
    )
