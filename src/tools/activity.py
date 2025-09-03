from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from src.tools.base import create_llm_chain
from langchain_core.tools import tool
import json
from typing import List

class PlanActivities(BaseModel):
    destination: str = Field(description="Selected destination")
    days: int = Field(description="Number of days")
    interest: str = Field(description="User's interest, e.g., 'relaxation', 'adventure'")
    activities: List[str] = Field(description="List of suggested activities")

    
activities_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. You plan activities for a trip based on user interests."),
    ("user", "Plan a day-by-day itinerary for a {days}-day trip to {destination}, focused on {interest}. "
             "Return a JSON list of activities, with a short description for each."),
])

chain = create_llm_chain(activities_prompt, structured=True, schema=PlanActivities)

@tool
def plan_activities(destination: str, interest: str = None, days: int = None) -> str:
    """
    Plans activities for a given destination and number of days based on user interest.

    Args:
        destination (str): The travel destination.
        interest (str): The user's interest, e.g., 'relaxation', 'adventure'.
        days (int): The number of days for the trip.

    Returns:
        str: JSON string containing the destination, interest, and planned activities.
    """
    # Check if the destination input is a JSON string containing both values
    try:
        data = json.loads(destination)
        if 'destination' in data and 'interest' in data and 'days' in data:
            destination = data['destination']
            interest = data['interest']
            days = data['days']
    except json.JSONDecodeError:
        pass  # It's not a JSON string, so we continue with the original values

    # Check if the interest value is valid before invoking the LLM chain
    # if not isinstance(interest, str) or not interest:
    #     raise ValueError("Interest must be a non-empty string.")

    # Validate that days is a positive integer
    # if not isinstance(days, int) or days <= 0:
    #     raise ValueError("Days must be a positive integer.")

    # Get a validated PlanActivities object
    result: PlanActivities = chain.invoke({"destination": destination, "days": days, "interest": interest})
    # Return as JSON string
    return result.model_dump_json()
