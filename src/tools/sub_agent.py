from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool,tool
from src.agent.structure import TravelItinerary
from src.tools.destination import recommend_destinations
from src.tools.cost import estimate_cost
from src.tools.activity import plan_activities
from src.agent.memory import create_memory
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv, find_dotenv
import json
# Load env
load_dotenv(find_dotenv())
api_key = os.environ.get("GOOGLE_API_KEY")

# Initialize LLM for sub-agent
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)

# Prompt guiding the sub-agent
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""
You are an itinerary planning assistant.

You have access to the following tools:
{tools}
Available tool names:
{tool_names}

Important:
1. When calling a tool, output exactly 2 lines:
   Action: <tool name>
   Action Input: <JSON-encoded string of input arguments>

2. When you have gathered enough information and are ready to answer,
   your FINAL ANSWER MUST be a single JSON object strictly matching this schema:

TravelItinerary:
- destination: str
- duration_days: int
- total_budget: float
- daily_plans: list of:
    - day: int
    - activities: list of str
    - estimated_cost: float

User request: {input}

{agent_scratchpad}
"""
)


tools = [
    Tool(name="Destination Recommender", func=recommend_destinations, description="Suggests travel destinations based on user preferences."),
    Tool(name="Cost Estimator", func=estimate_cost, description="Estimates total travel cost based on destination and duration."),
    Tool(name="Activity Planner", func=plan_activities, description="Recommends activities based on destination and interests."),
]


memory = create_memory(llm=llm, max_messages=5, summary_token_budget=250)

sub_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=sub_agent, tools=tools, memory=memory, verbose=True)
@tool
def generate_itinerary(input_data) -> TravelItinerary:
    """Generate a detailed travel itinerary based on preferences, budget, days, and interest."""
    # Safely handle both dict and JSON string inputs
    if isinstance(input_data, str):
        try:
            input_data = json.loads(input_data)
        except Exception as e:
            raise ValueError(f"Input must be a dict or valid JSON string: {e}")

    # Now input_data is a dictionary
    preference = input_data.get("preference")
    budget = input_data.get("budget")
    days = input_data.get("days")
    interest = input_data.get("interest")

    user_input = (
        f"Plan a trip:\n"
        f"- Preference: {preference}\n"
        f"- Budget: {budget}\n"
        f"- Days: {days}\n"
        f"- Interest: {interest}\n"
        f"Please generate a detailed day-by-day itinerary."
    )
    response = agent_executor.invoke({"input": user_input})
    output = response.get("output")

    try:
        if isinstance(output, str):
            itinerary = TravelItinerary.model_validate_json(output)
        else:
            itinerary = TravelItinerary.model_validate(output)
    except Exception:
        try:
            itinerary = TravelItinerary.model_validate(output)
        except Exception as e:
            raise ValueError(f"Failed to parse itinerary output: {e}")

    return itinerary