import json
from pydantic import BaseModel, Field, field_validator
from langchain.prompts import ChatPromptTemplate
from src.tools.base import create_llm_chain
from langchain_core.tools import tool

# Pydantic model for structured cost estimation
class CostEstimate(BaseModel):
    destination: str = Field(description="Selected destination")
    duration: int = Field(description="Number of days")
    total_cost: float = Field(description="Estimated total cost in USD")
    breakdown: dict = Field(description="Cost components")

    @field_validator('breakdown', mode='before')
    def parse_breakdown(cls, v):
        """
        Validator to handle different input formats for the breakdown.
        It attempts to parse the input as a JSON dictionary, a Python-style dictionary
        with single quotes, or a simple 'key: value' string.
        """
        if not v:
            raise ValueError("Breakdown must be a JSON dictionary string, but an empty string was provided.")

        # Return dicts as is (already valid)
        if isinstance(v, dict):
            return v

        if isinstance(v, str):
            try:
                # Try parsing as proper JSON first
                return json.loads(v)
            except json.JSONDecodeError:
                # Fallback to try fixing single quotes from Python-style dicts
                try:
                    fixed_v = v.replace("'", '"')
                    return json.loads(fixed_v)
                except json.JSONDecodeError:
                    # Final fallback: parse simple 'key: value' pairs
                    try:
                        parsed_dict = {}
                        pairs = v.split(',')
                        for pair in pairs:
                            key_value = pair.split(':', 1)
                            if len(key_value) == 2:
                                key = key_value[0].strip()
                                value = float(key_value[1].strip())
                                parsed_dict[key] = value
                        return parsed_dict
                    except Exception as e:
                        raise ValueError(f"Breakdown could not be parsed. Original error: {e}") from e
        else:
            raise ValueError("Breakdown must be a valid JSON dictionary string or a simple 'key: value' string.")

# Prompt to guide the LLM to output structured JSON
cost_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that estimates travel costs."),
    ("user", "Estimate travel cost for {duration} days at {destination}. "
             "Return a JSON with total_cost (float), breakdown (dict of hotel, flight, other costs).")
])

# Create structured output LLM chain tied to the CostEstimate schema
chain = create_llm_chain(cost_prompt, structured=True, schema=CostEstimate)

@tool
def estimate_cost(destination: str, duration: int = None) -> str:
    """
    Estimate the travel cost for a given destination and number of days.

    Args:
        destination (str): The travel destination (or a JSON string containing destination and days).
        duration (int): Number of days to stay.

    Returns:
        str: JSON string containing destination, days, total cost, and cost breakdown.
    """
    try:
        # Check if the input is a JSON string containing both destination and duration
        data = json.loads(destination)
        destination = data.get('destination', destination)
        duration = data.get('duration', duration)
    except (json.JSONDecodeError, TypeError):
        # If not a JSON string, proceed with the original inputs
        pass

    try:
        # Invoke the LLM chain to get the structured response
        result: CostEstimate = chain.invoke({"destination": destination, "duration": duration})
    except Exception as e:
        # Handle cases where the LLM chain fails to produce a valid response
        print(f"LLM chain failed to produce a valid response: {e}")
        result = CostEstimate(
            destination=destination,
            duration=duration,
            total_cost=0.0,
            breakdown={"error": "LLM failed to generate valid cost data. Default values provided."}
        )
    return result.model_dump_json(indent=2)
