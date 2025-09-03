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
import json

# Load environment variables
load_dotenv(find_dotenv())
api_key = os.environ.get("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)

# Define your tools
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
        func=generate_itinerary,
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
Important: When you call a tool, ALWAYS output exactly two lines in the following format:
Action: <tool name>
Action Input: <JSON-encoded string of input arguments>
Strictly follow this formatting.

Follow this reasoning process:
- Thought: I must consider what to do next. I must either provide a final answer or call a tool.
- If the user's query is missing key information (e.g., destination, duration, budget, or interests), provide a final answer that clearly and concisely asks for the missing information. DO NOT call a tool.
- If all necessary information is provided, call the appropriate tool.
- Action: pick a tool (if needed)
- Action Input: provide input for the tool, strictly as a JSON-encoded string
- Observation: record the result
- Repeat as needed until you can provide the final answer.

Final Answer: A final, single, concise answer to the user's question, or a question asking for more information.

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
    handle_parsing_errors=True,
    stop_sequence=['\nObservation:', '\nThought:']
)

# Updated entrypoint function to run the agent on user input
def run_agent(preference: str, budget: float, days: int, interest: str):
    """
    Run the travel planner agent with user preferences,
    budget, duration, and interest, and return structured output.
    """
    # Create a dictionary with the user's preferences
    user_preferences = {
        "preference": preference,
        "budget": budget,
        "days": days,
        "interest": interest,
    }

    # Format the input as a JSON string for a reliable handoff
    user_query = json.dumps(user_preferences)

    # Pass the formatted JSON string to the agent's input
    response = agent_executor.invoke({"input": user_query})

    return response.get("output")
