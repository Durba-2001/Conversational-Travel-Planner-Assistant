from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from src.tools.destination import recommend_destinations
from src.tools.cost import estimate_cost
from src.tools.activity import plan_activities
from src.tools.sub_agent import generate_itinerary
from src.agent.memory import create_memory
from src.agent.structure import TravelItinerary
import os
from dotenv import load_dotenv, find_dotenv
import json

# Load env vars
load_dotenv(find_dotenv())
api_key = os.environ.get("GOOGLE_API_KEY")

# Initialize LLM and wrap with structured output schema binding
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)
structured_llm = llm.with_structured_output(TravelItinerary)

# Define tools
tools = [
    Tool(
        name="Destination Recommender",
        func=recommend_destinations,
        description="Suggests travel destinations based on user preferences."
    ),
    Tool(
        name="Cost Estimator",
        func=estimate_cost,
        description="Estimates the total travel cost based on destination and duration."
    ),
    Tool(
        name="Activity Planner",
        func=plan_activities,
        description="Plans activities for the trip considering destination and interests."
    ),
    Tool(
        name="Itinerary Generator",
        func=generate_itinerary,  # This should invoke structured_llm internally
        description="Generates a day-by-day travel itinerary."
    ),
]

prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""
You are a thoughtful travel planning assistant.

You have access to these tools:
{tools}

Available tool names:
{tool_names}

When calling a tool, ALWAYS output exactly 2 lines:
Action: <tool name>
Action Input: <JSON-encoded string of input arguments>

When ready to answer, output a single JSON matching this schema:
{{
  "destination": "string",
  "duration": "integer",
  "total_budget": "float",
  "daily_plans": [
    {{
      "duration": "integer",
      "activities": ["string"],
      "estimated_cost": "float"
    }}
  ]
}}

If missing data, fill fields with null or defaults, do NOT output free text.

User question: {input}

{agent_scratchpad}
"""
)

# Create memory and agent
conversation_memory = create_memory(llm=llm, max_messages=10, summary_token_budget=300)
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=conversation_memory,
    verbose=True,
    max_iterations=4,
    early_stopping_method="force",
    handle_parsing_errors=True,  # Enable auto-retry on output parsing errors
    stop_sequence=['\nObservation:', '\nThought:']
)

# Entrypoint
def run_agent(preference: str, budget: float, duration: int, interest: str) -> TravelItinerary:
    user_preferences = {
        "preference": preference,
        "budget": budget,
        "duration": duration,
        "interest": interest,
    }
    user_query = json.dumps(user_preferences)
    
    # Use the reactive agent executor to invoke and get structured output
    response = agent_executor.invoke({"input": user_query})
    
    # The `output` will be a parsed TravelItinerary instance due to `structured_llm` binding
    return response.get("output")
