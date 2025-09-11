## Conversational Travel Planner Assistant::

This project involves building a conversational AI assistant using the LangChain framework to help users plan personalized travel itineraries. The assistant will interact via natural language to gather preferences (e.g., destination, budget, interests like adventure or relaxation) and generate itineraries. It will use advanced LangChain features for memory management, structured outputs, and tool integration, with a FastAPI backend and MongoDB for persistent storage. This is an educational project for students to learn modern AI agent development with a production-ready structure.

The assistant will:

- Recommend destinations and generate daily itineraries with activities and costs.
- Remember recent conversations (last 5 messages) and summarize older ones.
- Use tools, including a sub-agent, for tasks like destination suggestion or itinerary generation.
- Expose secure FastAPI endpoints for interaction.

### Tech Stack

- **Core Framework**: LangChain (latest version).
- **LLM**: Gemini
- **Backend**: FastAPI, Uvicorn.
- **Database**: MongoDB (via `pymongo`).
- **Libraries**: Pydantic (structured outputs), PyJWT (auth), Bcrypt (hashing), python-dotenv.
- **Testing**: Pytest for unit and integration tests.

### File Structure
<pre>
travel_planner/
├── src/
│   ├── __init__.py
│   ├── main.py                # FastAPI app entry point
│   ├── config.py              # Environment variables (pydantic-settings)
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── models.py          # Pydantic models for auth
│   │   └── router.py          # Auth endpoints
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── core.py            # Main agent setup
│   │   ├── memory.py          # Custom memory with summarization
│   │   └── structured.py      # Output parsers
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py            # Tool utilities
│   │   ├── destination.py     # Destination recommender tool
│   │   ├── cost.py            # Cost estimator tool
│   │   ├── activity.py        # Activity planner tool
│   │   └── sub_agent.py       # Itinerary sub-agent
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py          # MongoDB schemas
│   │   └── session.py         # MongoDB connections
│   └── api/
│       ├── __init__.py
│       └── router.py          # Chat endpoints
├── tests/
│   ├── unit/
│   │   └── test_tools.py      # Unit tests
│   └── integration/
│       └── test_api.py        # API tests
├── .env.example               # Sample env file
├── pyproject.toml             # Black, flake8 config
└── README.md                  # Setup instructions
