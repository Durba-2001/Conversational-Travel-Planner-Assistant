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

class DestinationSuggestion(BaseModel):
    destinations: List[str] = Field(description="List of recommended destinations")
    reasoning: str = Field(description="Explanation for suggestions")
    
class PlanActivities(BaseModel):
    destination: str
    duration: int
    interest: str
    budget: float
    daily_plans: List[DailyPlan]

class CostEstimate(BaseModel):
    destination: str = Field(description="Selected destination")
    days: int = Field(description="Number of days")
    total_cost: float = Field(description="Estimated cost in USD")
    breakdown: dict = Field(description="Cost components")

