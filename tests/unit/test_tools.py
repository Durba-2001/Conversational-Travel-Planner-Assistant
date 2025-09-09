import pytest
import json
from unittest.mock import patch
from src.tools.activity import plan_activities
from src.tools.sub_agent import generate_itinerary
from src.tools.destination import recommend_destinations
from src.tools.cost import estimate_cost

# ---------------------------
# 1. Test plan_activities
# ---------------------------

@patch("src.tools.activity.chain.invoke")
def test_plan_activities_mocked(mock_invoke):
    mock_invoke.return_value.model_dump_json.return_value = json.dumps({
        "destination": "Paris",
        "duration": 2,
        "interest": "sightseeing",
        "budget": 2000,
        "daily_plans": [
            {"duration": 1, "activities": ["Eiffel Tower"], "estimated_cost": 500},
            {"duration": 2, "activities": ["Louvre Museum"], "estimated_cost": 400}
        ]
    })
    
    result_json = plan_activities(destination="Paris", interest="sightseeing", duration=2, budget=2000)
    result = json.loads(result_json)
    assert result["destination"] == "Paris"
    assert len(result["daily_plans"]) == 2
    assert result["daily_plans"][0]["duration"] == 1

# ---------------------------
# 2. Test estimate_cost
# ---------------------------
def test_estimate_cost_basic():
    inputs = {"destination": "Rome", "days": 3}
    result_json = estimate_cost(inputs)
    result = json.loads(result_json)
    assert result["destination"] == "Rome"
    assert result["days"] == 3
    assert result["total_cost"] == 1800  # 3*500 + 300
    assert "hotel" in result["breakdown"]
    assert "flight" in result["breakdown"]

# ---------------------------
# 3. Test recommend_destinations
# ---------------------------

@patch("src.tools.destination.chain.invoke")
def test_recommend_destinations_mocked(mock_invoke):
    mock_invoke.return_value.model_dump_json.return_value = json.dumps({
        "destinations": ["Paris", "Rome", "Venice"],
        "reasoning": "Based on budget and preference"
    })
    
    result_json = recommend_destinations(preference="historical", budget=3000)
    result = json.loads(result_json)
    assert "destinations" in result
    assert len(result["destinations"]) == 3

# ---------------------------
# 4. Test generate_itinerary
# ---------------------------

@patch("src.tools.sub_agent.ChatGoogleGenerativeAI.invoke")
def test_generate_itinerary_mocked(mock_invoke):
    mock_response = json.dumps({
        "destination": "Paris",
        "duration_days": 2,
        "total_budget": 2000,
        "daily_plans": [
            {"day": 1, "activities": ["Eiffel Tower"], "estimated_cost": 500},
            {"day": 2, "activities": ["Louvre Museum"], "estimated_cost": 400}
        ]
    })
    mock_invoke.return_value.content = mock_response

    result = generate_itinerary("Plan a 2-day trip to Paris")
    assert result.destination == "Paris"
    assert result.duration == 2
    assert len(result.daily_plans) == 2
    assert result.daily_plans[0].duration == 1
    assert result.daily_plans[1].activities == ["Louvre Museum"]
