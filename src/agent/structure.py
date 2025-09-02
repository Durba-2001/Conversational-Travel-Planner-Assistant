from pydantic import BaseModel, Field
from typing import List

class DayPlan(BaseModel):
    day: int = Field(description="Day number")
    activities: List[str] = Field(description="List of activities")
    estimated_cost: float = Field(description="Cost in USD")

class TravelItinerary(BaseModel):
    destination: str = Field(description="Travel destination")
    duration_days: int = Field(description="Total days")
    total_budget: float = Field(description="Estimated total budget in USD")
    daily_plans: List[DayPlan] = Field(description="Breakdown by day")
