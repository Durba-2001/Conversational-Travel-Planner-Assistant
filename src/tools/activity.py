from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from src.tools.base import create_llm_chain
from langchain_core.tools import tool
from typing import List
from src.agent.structure import ActivityList  


# Prompt for activity recommendations (not daily plans)
activities_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Return JSON matching the schema."),
    ("user", "Suggest activities for {destination} based on the user's interest in {interest}. "
             "Return a list of recommended activities (at least 5).")
])


chain = create_llm_chain(activities_prompt, structured=True, schema=ActivityList)

@tool
def plan_activities(destination: str, interest: str = None) -> str:
    """
    Suggest activities for a given destination based on user interest.

    Args:
        destination (str): Selected travel destination.
        interest (str, optional): User's interest, e.g., 'relaxation', 'adventure'.

    Returns:
        str: JSON string containing destination, interest, and a list of recommended activities.
    """
    result: ActivityList = chain.invoke(
        {"destination": destination, "interest": interest}
    )
    return result.model_dump_json(indent=2)
