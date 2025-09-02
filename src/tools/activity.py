from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List
from src.tools.base import create_llm_chain, safe_parse_json
from langchain.prompts import ChatPromptTemplate

class ActivityList(BaseModel):
    destination: str = Field(description="Selected destination")
    interest: str = Field(description="User's interest")
    activities: List[str] = Field(description="Recommended activities")

activity_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a travel assistant that suggests activities."),
    ("user", "List 5 activities for {interest} in {destination}. Return the answer in JSON.")
])

chain = create_llm_chain(activity_prompt)

@tool
def plan_activities(destination: str, interest: str)-> ActivityList:
    """
    Suggests activities for a given destination and user interest.

    Args:
        destination (str): The travel destination.
        interest (str): The user's interest (e.g., adventure, history, food).

    Returns:
        ActivityList: Includes the destination, interest, and a list of recommended activities.
    """
    result = chain.invoke({"destination": destination, "interest": interest})
    parsed = safe_parse_json(result)
    activities = parsed.get("activities", []) 
    return ActivityList(destination=destination, interest=interest, activities=activities)
