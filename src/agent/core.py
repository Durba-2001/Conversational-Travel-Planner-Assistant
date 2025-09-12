from src.config import api_key
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

from src.agent.memory import get_agent_with_memory
from src.tools.destination import recommend_destinations
from src.tools.cost import estimate_cost
from src.tools.activity import plan_activities
from src.tools.sub_agent import generate_itinerary
from src.agent.structure import TravelItinerary


# --- LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=api_key)

# --- Tools ---
tools = [
    Tool("recommend_destinations", recommend_destinations, "Suggests destinations."),
    Tool("estimate_cost", estimate_cost, "Estimates total travel cost."),
    Tool("plan_activities", plan_activities, "Plans day-by-day activities."),
    Tool(
        "generate_itinerary",
        generate_itinerary,
        "MANDATORY FINAL STEP: Combines destinations, costs, and activities into one structured TravelItinerary before giving the final answer."
    )
]

# --- Prompt ---
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""
You are a friendly and polite human-like travel assistant. 

Guidelines for your behavior:
- You ONLY answer travel-related questions (destinations, itineraries, activities, flights, hotels, budgets, etc.). 
- If the user asks something unrelated to travel, DO NOT use Thought/Action/Observation.  
  Instead, directly give a polite refusal with only a Final Answer:: encouraging them to ask about travel.   
- If the user does not mention the number of days for the trip, assume it is a 1-day trip.  
- Always provide your response in a natural, conversational, and professional tone, as if speaking directly to the traveler.  
You have access to the following tools:

{tools}
Tool names: {tool_names}

Use the following format:

Thought: ...
Action: ...
Action Input: ...
Observation: ...
Thought: I now know the final answer
Final Answer: detailed human-readable plan

User request: {input}

{agent_scratchpad}
"""
)

# --- Agent ---
react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=react_agent,
    tools=tools,
    verbose=True,
    max_execution_time=120,
    handle_parsing_errors=True
)


# --- Run agent ---
def run_agent(message: str, session_id: str) -> str:
    agent_with_memory = get_agent_with_memory(agent_executor)

    response = agent_with_memory.invoke(
        {"input": message, "tool_names": [t.name for t in tools], "tools": tools},
        config={"configurable": {"session_id": session_id}}
    )

    # --- Handle TravelItinerary output ---
    if isinstance(response, TravelItinerary):
        text = f"Your {response.duration}-day travel plan to {response.destination} within budget ${response.total_budget}. "
        text += "Daily activities: "
        total_cost = 0
        for day_plan in response.daily_plans:
            day = day_plan.get("day", "?")
            activities = ", ".join(day_plan.get("activities", []))
            cost = day_plan.get("estimated_cost", 0)
            total_cost += cost
            text += f"Day {day}: {activities}. Cost: ${cost}. "
        text += f"Total estimated activity cost: ${total_cost}."
    elif isinstance(response, dict) and "output" in response:
        text = str(response["output"])
    else:
        text = str(response)

    return " ".join(text.replace("\n", " ").replace("*", "").split())
