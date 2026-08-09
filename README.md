# TripPilot AI

> A real-time, multi-agent travel planner that turns a natural-language trip brief into a reviewable itinerary.

TripPilot AI coordinates specialized agents for flights, accommodation, weather, budget, and itinerary design. A LangGraph supervisor validates and routes each request, while a human approval step keeps the traveler in control before a final plan is produced.

## Highlights

- **Multi-agent orchestration** — a supervisor dynamically chooses the specialists required for each trip.
- **Guardrailed requests** — validates relevance, safety, and request quality before research begins.
- **Live progress updates** — the frontend consumes Server-Sent Events (SSE) to show agent activity as it happens.
- **Human-in-the-loop review** — approve a draft or request targeted revisions before the final response.
- **MCP-connected research** — integrates Tavily, AviationStack, and a custom weather server through the Model Context Protocol.
- **Durable conversations** — LangGraph checkpoints are stored in PostgreSQL, so a plan can be resumed safely.
- **Exportable result** — copy an itinerary or download it as a PDF from the web interface.

## System flow

```text
Traveler request
       |
Input guardrail
       |
LangGraph supervisor
       |
       +--> Flight agent      (AviationStack MCP)
       +--> Hotel agent       (Tavily MCP)
       +--> Weather agent     (Custom Weather MCP)
       +--> Budget agent      (LLM analysis)
       +--> Itinerary agent   (LLM synthesis)
       |
Draft itinerary --> Human review --> Final response
       |
PostgreSQL checkpointing and conversation state
```

## Technology

| Area | Tools |
| --- | --- |
| API and web app | FastAPI, Uvicorn, Jinja2 |
| Orchestration | LangGraph, LangChain |
| Language model | Groq (`llama-3.3-70b-versatile`) |
| Agent integrations | Model Context Protocol (MCP), `langchain-mcp-adapters` |
| Research and travel data | Tavily, AviationStack, OpenWeather |
| Persistence | PostgreSQL, `langgraph-checkpoint-postgres` |
| Observability | LangSmith tracing (configured through environment variables) |
| Frontend | Vanilla JavaScript, HTML/CSS, SSE, Marked, html2pdf.js |

## Project structure

```text
.
├── app.py                       # FastAPI routes, page rendering, SSE endpoints
├── backend.py                   # LangGraph state, agents, routing, HITL and streams
├── mcp_client.py                # MCP client configuration and travel-data helpers
├── custom_weather_mcp_server.py # Local FastMCP weather server
├── templates/
│   └── index.html               # Web application markup
├── static/
│   ├── style.css                # Product UI styles
│   └── script.js                # Streaming UI, agent progress, export interactions
├── requirements.txt
└── .env                         # Local credentials (never commit this file)
```

## Prerequisites

- Python 3.10 or newer
- PostgreSQL database reachable from your development environment
- [uv](https://docs.astral.sh/uv/) / `uvx` available on `PATH` for the AviationStack MCP server
- API credentials for Groq, Tavily, AviationStack, and OpenWeather

## Setup

### 1. Clone and create a virtual environment

```powershell
git clone <your-repository-url>
cd Trip-Planner-Multi-Agent-System
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate with `source .venv/bin/activate`.

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the repository root:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
AVIATIONSTACK_API_KEY=your_aviationstack_key
OPENWEATHER_API_KEY=your_openweather_key
DATABASE_URL=postgresql://user:password@host:5432/database
DEFAULT_ORIGIN_IATA=DAC

# Optional LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=trippilot-ai
```

`backend.py` adds `sslmode=require` to the database URL when it is not already present. For local PostgreSQL, supply the SSL mode appropriate for your setup.

### 4. Start the application

```powershell
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The custom weather MCP server is launched by the configured MCP client when needed. It can also be run independently during development:

```powershell
python custom_weather_mcp_server.py
```

## Using the app

1. Enter a request such as: `Plan a 7-day Japan trip from Bangladesh with flights, hotels, and sightseeing under 2 lakhs.`
2. Watch the live execution view as the supervisor routes work to specialist agents.
3. Review the generated draft itinerary.
4. Approve it to create the final plan, or provide revision feedback to run the final refinement step.
5. Copy the plan or save it as a PDF.

## API reference

### Create a travel plan

`POST /api/travel`

```json
{
  "message": "Plan a 5-day Dubai trip from Dhaka with flights and hotels.",
  "thread_id": "optional-existing-thread-id"
}
```

Returns a complete JSON response. Use this endpoint for non-streaming clients.

### Stream a travel plan

`POST /api/travel/stream`

Accepts the same body as `/api/travel` and responds with an `text/event-stream` sequence. Events include `start`, `node_complete`, `interrupt`, `complete`, and `error`.

### Approve or revise a draft

`POST /api/travel/approve`

```json
{
  "thread_id": "travel-thread-id",
  "approved": true,
  "feedback": ""
}
```

Set `approved` to `false` and provide feedback to request changes.

### Stream approval or revision

`POST /api/travel/approve/stream`

Accepts the same approval payload and emits SSE progress updates while the workflow resumes.

### Health check

`GET /health`

Returns the API status and enabled workflow capabilities.

## Development notes

- The workflow state is modeled as `TravelState` in `backend.py`.
- The supervisor uses structured LLM output to select a subset of the available agents for a request.
- The human approval node uses LangGraph interruption and resume semantics.
- `nest_asyncio` is used so synchronous convenience wrappers can safely invoke async MCP helpers in the FastAPI process.
- Keep `.env` private. The repository’s `.gitignore` should exclude it before any push.

## License

Add a license file that matches your intended distribution model before publishing or deploying this project.
