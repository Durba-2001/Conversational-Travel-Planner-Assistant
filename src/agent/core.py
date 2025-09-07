import os
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.agent.memory import get_session_memory  # sync memory
from src.tools.destination import recommend_destinations
from src.tools.cost import estimate_cost
from src.tools.activity import plan_activities
from src.tools.sub_agent import generate_itinerary
from src.agent.structure import TravelItinerary

# Load API key
load_dotenv(find_dotenv())
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)

# Define tools
tools = [
    Tool("Destination Recommender", recommend_destinations, "Suggests destinations."),
    Tool("Cost Estimator", estimate_cost, "Estimates total travel cost."),
    Tool("Activity Planner", plan_activities, "Plans day-by-day activities."),
    Tool("Generate Itinerary", generate_itinerary, "Generates full travel itinerary.")
]

# Prompt template
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""
You are a travel assistant. You have access to the following tools:

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

# Build agent
react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=react_agent,
    tools=tools,
    verbose=True,
    max_iterations=20,
    max_execution_time=120,
    handle_parsing_errors=True
)

# Wrap agent with memory
def get_agent_with_memory(session_id: str):
    memory = get_session_memory(session_id)  # sync memory
    return RunnableWithMessageHistory(
        agent_executor,
        get_session_history=lambda _: memory.chat_memory
    )

# --- Synchronous run_agent ---
def run_agent(message: str, session_id: str) -> str:
    """
    Run the travel agent synchronously and return a human-readable string.
    """
    memory = get_session_memory(session_id)
    response = agent_executor.invoke(
        {"input": message, "tool_names": [t.name for t in tools], "tools": tools},
        config={"configurable": {"session_id": session_id, "memory": memory}}
    )

    # Handle TravelItinerary objects
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

    # Flatten text
    text = text.replace("\n", " ").replace("*", "")
    text = " ".join(text.split())
    return text
