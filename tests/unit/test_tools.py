import pytest
import json
from src.tools.destination import recommend_destinations
from src.tools.cost import estimate_cost
from src.tools.activity import plan_activities
from src.tools.sub_agent import generate_itinerary


# ----------------- Helper ----------------- #
import json
from pydantic import BaseModel

import json
from pydantic import BaseModel

def normalize_result(result):
    """Ensure tool output is always a dict."""
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return {"error": "Invalid JSON output", "raw": result}
    elif isinstance(result, BaseModel):  # Works for TravelItinerary
        return result.model_dump()
    elif isinstance(result, dict):
        return result
    else:
        return {"error": f"Unsupported result type: {type(result)}", "raw": str(result)}




# ----------------- Destination Recommender ----------------- #
def test_recommend_destination_valid():
    inputs = {"preference": "beach, relaxing", "budget": 1000}
    result = normalize_result(recommend_destinations.invoke(inputs))

    assert isinstance(result, dict)
    assert "destinations" in result
    assert isinstance(result["destinations"], list)
    assert len(result["destinations"]) > 0


def test_recommend_destination_invalid():
    inputs = {"preference": "", "budget": 0}
    result = normalize_result(recommend_destinations.invoke(inputs))

    assert isinstance(result, dict)
    assert "destinations" in result
    assert isinstance(result["destinations"], list)
    # Expect fallback or free destinations
    assert len(result["destinations"]) >= 1


# ----------------- Cost Estimator ----------------- #
def test_estimate_cost_valid():
    inputs = {"inputs": {"destination": "Paris", "days": 5}}
    result = normalize_result(estimate_cost.invoke(inputs))

    assert isinstance(result, dict)
    assert "total_cost" in result
    assert result["total_cost"] > 0
    assert result["destination"] == "Paris"



def test_estimate_cost_invalid():
    inputs = {"inputs": {"destination": "", "days": -1}}
    result = normalize_result(estimate_cost.invoke(inputs))

    assert isinstance(result, dict)
    assert "total_cost" in result
    # Invalid input → cost should not be positive
    assert result["total_cost"] <= 0 or "error" in result




# ----------------- Activity Planner ----------------- #
def test_plan_activities_valid():
    inputs = {"destination": "Goa", "duration": 3, "interest": "adventure", "budget": 500}
    result = normalize_result(plan_activities.invoke(inputs))

    assert isinstance(result, dict)
    assert "destination" in result
    assert result["destination"] == "Goa"
    assert "duration" in result
    assert result["duration"] == 3
    assert "interest" in result
    assert result["interest"] == "adventure"
    assert "budget" in result
    assert result["budget"] > 0



def test_plan_activities_invalid():
    inputs = {"destination": "", "duration": 0, "interest": "", "budget": -100}
    result = normalize_result(plan_activities.invoke(inputs))

    assert isinstance(result, dict)
    # At least one field should signal invalid state
    assert result.get("destination", "") == "" or result.get("duration", 0) <= 0


# ----------------- Itinerary Sub-Agent ----------------- #
def test_generate_itinerary_valid():
    result = normalize_result(generate_itinerary.invoke("Plan a 2-day trip to Goa"))

    assert isinstance(result, dict)
    assert "destination" in result
    assert "duration" in result
    assert "daily_plans" in result



def test_generate_itinerary_invalid():
    raw = generate_itinerary.invoke("")     # Returns TravelItinerary
    result = normalize_result(raw)          # Convert to dict

    assert isinstance(result, dict)
    assert "error" in result or result.get("destination", "") in ("", "Unknown")

