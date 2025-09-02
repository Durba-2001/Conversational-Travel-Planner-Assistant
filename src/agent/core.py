# src/agent/core.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from src.tools.destination import recommend_destinations
from src.tools.activity import plan_activities
from src.tools.cost import estimate_cost
from src.tools.sub_agent import generate_itinerary
from src.agent.memory import create_memory
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
api_key = os.environ.get("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)

# Define your tools as LangChain Tool objects wrapping your functions
tools = [
    Tool(
        name="Destination Recommender",
        func=recommend_destinations,
        description="Suggests travel destinations based on user preferences."
    ),
    Tool(
        name="Cost Estimator",
        func=estimate_cost,
        description="Estimates the total travel cost based on destination and activities."
    ),
    Tool(
        name="Activity Planner",
        func=plan_activities,
        description="Plans activities for the trip considering destination and duration."
    ),
    Tool(
        name="Itinerary Generator",
        func= generate_itinerary,
        description="Generates a day-by-day travel itinerary."
    ),
]

# Prompt template for the agent
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""You are a helpful travel planning assistant.

You have access to the following tools:
{tools}

The available tool names are:
{tool_names}
Important: When calling a tool, ALWAYS include all required arguments 
exactly as specified in its schema.
Follow this reasoning process:
- Thought: consider what to do next
- Action: pick a tool (if needed)
- Observation: record the result
- Repeat as needed until you can provide the final answer.

User question: {input}

{agent_scratchpad}"""
)

# Create memory to store last 10 messages with summarization budget
conversation_memory = create_memory(llm=llm, max_messages=10, summary_token_budget=300)

# Create the agent using react agent (tool-calling with prompt & tools)
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Create AgentExecutor to run the agent with tools and memory
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=conversation_memory,
    verbose=True,
    handle_parsing_errors=True  
)

# Entrypoint function to run the agent on user input
def run_agent(preference: str, budget: float, days: int, interest: str):
    """
    Run the travel planner agent with user preferences,
    budget, duration, and interest, and return structured output.
    """
    user_input = (
        f"Plan a {preference} trip with a budget of ${budget} for {days} days "
        f"focusing on {interest} activities."
    )
    # Run the agent_executor on the formatted user input
    response = agent_executor.invoke({"input": user_input})
    return response.get("output")
