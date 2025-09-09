# src/tools/activity.py
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from src.tools.base import create_llm_chain
from langchain_core.tools import tool
from typing import List
import json

class DailyPlan(BaseModel):
    duration: int
    activities: List[str]
    estimated_cost: float

class PlanActivities(BaseModel):
    destination: str
    duration: int
    interest: str
    budget: float
    daily_plans: List[DailyPlan]

activities_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Return JSON matching the schema."),
    ("user", "Plan a {duration}-day itinerary for {destination} focused on {interest}.")
])

chain = create_llm_chain(activities_prompt, structured=True, schema=PlanActivities)

@tool
def plan_activities(destination: str, interest: str = None, duration: int = 1, budget: float = None) -> str:
    """
    Plan day-by-day activities for a given destination based on user interest, duration, and budget.

    Args:
        destination (str): Selected travel destination.
        interest (str, optional): User's interest, e.g., 'relaxation', 'adventure'.
        duration (int, optional): Number of days for the trip.
        budget (float, optional): Total budget for the trip.

    Returns:
        str: JSON string containing destination, interest, duration, budget, and daily activity plans.
    """
    result: PlanActivities = chain.invoke(
        {"destination": destination, "duration": duration, "interest": interest, "budget": budget}
    )
    return result.model_dump_json(indent=2)
