from pydantic import BaseModel, Field
from typing import List


class DailyPlan(BaseModel):
    duration: int = Field(description="Day number of the trip")
    activities: List[str] = Field(description="List of activities for this day")
    estimated_cost: float = Field(description="Estimated cost for the day")

class TravelItinerary(BaseModel):
    destination: str = Field(description="Travel destination")
    duration: int = Field(description="Total days")
    total_budget: float = Field(description="Estimated total budget in USD")
    daily_plans: List[DailyPlan] = Field(description="Breakdown by day")
