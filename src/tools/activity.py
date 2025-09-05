from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from src.tools.base import create_llm_chain
from langchain_core.tools import tool
from typing import List
import json


class DailyPlan(BaseModel):
    duration: int = Field(description="Day number of the trip")
    activities: List[str] = Field(description="List of activities for this day")
    estimated_cost: float = Field(description="Estimated cost for the day")



class PlanActivities(BaseModel):
    destination: str = Field(description="Selected destination")
    duration: int = Field(description="Number of days")
    interest: str = Field(description="User's interest, e.g., 'relaxation', 'adventure'")
    budget: float = Field(description="Estimated total budget for the trip")
    daily_plans: List[DailyPlan] = Field(description="Day-by-day activity breakdown")


activities_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Always return JSON matching the schema."),
    ("user", "Suggest a {duration}-day itinerary for {destination}, focused on {interest}. "
             "Provide a JSON object with a 'daily_plans' list, where each item contains "
             "the day number and a list of activities.")
])


chain = create_llm_chain(activities_prompt, structured=True, schema=PlanActivities)


@tool
def plan_activities(destination: str, interest: str = None, duration: int = None) -> str:
    """
    Plans activities for a given destination and number of days based on user interest.

    Args:
        destination (str): The travel destination.
        interest (str): The user's interest, e.g., 'relaxation', 'adventure'.
        duration (int): The number of days for the trip.

    Returns:
        str: JSON string containing the destination, interest, and day-by-day planned activities.
    """
    try:
        data = json.loads(destination)
        if 'destination' in data and 'interest' in data and 'duration' in data:
            destination = data['destination']
            interest = data['interest']
            duration = data['duration']
    except json.JSONDecodeError:
        pass

    result: PlanActivities = chain.invoke(
        {"destination": destination, "duration": duration, "interest": interest}
    )
    return result.model_dump_json(indent=2)
