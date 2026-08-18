import os
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
    client as mcp_client,
)


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#if not GROQ_API_KEY:
#    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Please add it to your .env file.")

# =========================
# LLM - original model kept
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
)

# =========================
# State - original fields kept, new control fields added
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent
# =========================
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Live/Reference AviationStack Data:
{flight_data}

Instructions:
- If valid AviationStack flight-search data is present, use it as the primary
  source for flight recommendations.
- Clearly distinguish live/API-derived information from general estimates.
- Never invent live flight numbers, live departure times, live arrival times,
  or live flight statuses.
- If flight search is unavailable, incompatible, historical-only, or returns
  an error, provide useful generic flight-planning guidance from your general
  knowledge and clearly label it as estimated/general guidance.
- Do not treat airport/airline reference lists as proof that a route operates.
- If the origin is a country rather than a specific airport, say that a major
  gateway must be selected instead of inventing one.
- Airport and airline reference data may be used to identify likely airport
  codes and airlines, but it is not itself proof that a specific flight is
  operating.

Generate concise guidance covering:
1. Departure and arrival airports
2. Airlines/routes when supported by the data
3. Typical flight duration
4. Estimated airfare range when live pricing is unavailable
5. Peak-season considerations
6. Booking advice
"""


def _extract_text_from_mcp(result: Any) -> str:
    """Convert common MCP content responses into searchable text."""
    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(result)


def _mcp_payload_has_error(result: Any) -> bool:
    """Detect API/auth errors even when MCP returns them without raising."""
    raw = _extract_text_from_mcp(result).lower()

    return any(
        marker in raw
        for marker in (
            '"ok": false',
            "invalid_access_key",
            "api_error",
            "unauthorized",
            "authentication",
            '"error"',
        )
    )


def _parse_json_payload(result: Any) -> Any:
    """Best-effort extraction of JSON returned inside MCP text content."""
    raw = _extract_text_from_mcp(result).strip()

    if not raw:
        return None

    try:
        return json.loads(raw)
    except Exception:
        pass

    # MCP sometimes wraps JSON in surrounding text.
    first_obj = raw.find("{")
    last_obj = raw.rfind("}")
    if first_obj != -1 and last_obj > first_obj:
        try:
            return json.loads(raw[first_obj:last_obj + 1])
        except Exception:
            pass

    first_arr = raw.find("[")
    last_arr = raw.rfind("]")
    if first_arr != -1 and last_arr > first_arr:
        try:
            return json.loads(raw[first_arr:last_arr + 1])
        except Exception:
            pass

    return raw


def _airport_code_from_reference(
    reference_result: Any,
    location: str,
) -> str | None:
    """
    Resolve a city/country/airport name to an IATA code using the
    AviationStack airport reference response.
    """
    if not location:
        return None

    location = str(location).strip().lower()

    # AviationStack reference endpoints can return a limited/default page.
    # Resolve common explicit airport/city names safely as a code convenience.
    # This does NOT mean a flight on that route is live or available.
    common_iata = {
        "hyderabad airport": "HYD",
        "hyderabad": "HYD",
        "rajiv gandhi international airport": "HYD",
        "paris charles de gaulle airport": "CDG",
        "charles de gaulle": "CDG",
        "cdg": "CDG",
        "paris orly airport": "ORY",
        "orly": "ORY",
        "ory": "ORY",
        "dhaka": "DAC",
        "hazrat shahjalal international airport": "DAC",
        "dubai": "DXB",
        "dubai international airport": "DXB",
        "bangkok": "BKK",
        "suvarnabhumi airport": "BKK",
        "phuket": "HKT",
        "phuket international airport": "HKT",
    }

    if location in common_iata:
        return common_iata[location]

    payload = _parse_json_payload(reference_result)

    if not isinstance(payload, list):
        return None

    # Prefer exact IATA/city-name matches, then partial matches.
    candidates = []

    for airport in payload:
        if not isinstance(airport, dict):
            continue

        airport_name = str(airport.get("airport_name") or "").lower()
        iata = str(airport.get("iata_code") or "").strip().upper()
        city_iata = str(airport.get("city_iata_code") or "").strip().upper()
        country_name = str(airport.get("country_name") or "").lower()

        if not iata:
            continue

        score = 0

        if location == airport_name:
            score += 100
        if location == iata.lower():
            score += 100
        if location == city_iata.lower():
            score += 100
        if location == country_name:
            score += 20
        if location in airport_name:
            score += 50
        if location in country_name:
            score += 20

        if score:
            candidates.append((score, iata))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


async def _find_aviation_flight_tool():
    """
    Discover the actual flight-search tool exposed by the installed
    aviationstack-mcp package instead of hard-coding a tool name.
    """
    tools = await mcp_client.get_tools(server_name="aviationstack")

    # Never treat airport/airline reference tools as flight-search tools.
    candidates = []

    for tool in tools:
        name = str(getattr(tool, "name", "")).lower()
        description = str(getattr(tool, "description", "")).lower()
        searchable = f"{name} {description}"

        # Historical/date-reference tools are not generic live route-search
        # tools. If the installed MCP package exposes only such a tool, let
        # the existing LLM fallback handle the flight guidance.
        is_historical = any(
            word in name
            for word in (
                "historical",
                "by_date",
                "bydate",
            )
        )

        if (
            any(
                word in searchable
                for word in (
                    "flight",
                    "route",
                    "schedule",
                    "departure",
                    "arrival",
                )
            )
            and not any(
                word in name
                for word in (
                    "airport",
                    "airline",
                )
            )
            and not is_historical
        ):
            candidates.append(tool)

    if not candidates:
        return None

    # Prefer names that explicitly look like flight retrieval/search tools.
    priority = (
        "search_flights",
        "get_flights",
        "flight_search",
        "list_flights",
        "flights",
        "flight",
    )

    for preferred in priority:
        for tool in candidates:
            if preferred in str(getattr(tool, "name", "")).lower():
                return tool

    return candidates[0]


def _build_flight_tool_args(
    tool: Any,
    origin_code: str | None,
    destination_code: str | None,
) -> dict[str, Any]:
    """
    Build arguments from the actual discovered tool schema.

    Different MCP versions may use origin/destination, dep_iata/arr_iata,
    departure_iata/arrival_iata, etc. This keeps backend.py resilient without
    assuming one exact schema.
    """
    schema = getattr(tool, "args_schema", None)
    fields = getattr(schema, "model_fields", None)

    if fields is None:
        fields = getattr(schema, "__fields__", None)

    if not fields:
        return {}

    args: dict[str, Any] = {}

    for field_name in fields:
        name = str(field_name).lower()

        if any(
            token in name
            for token in (
                "dep_iata",
                "departure_iata",
                "origin_iata",
                "source_iata",
                "from_iata",
                "dep_airport",
                "origin_airport",
                "departure_airport",
            )
        ):
            if origin_code:
                args[field_name] = origin_code

        elif any(
            token in name
            for token in (
                "arr_iata",
                "arrival_iata",
                "destination_iata",
                "dest_iata",
                "to_iata",
                "arr_airport",
                "destination_airport",
                "arrival_airport",
            )
        ):
            if destination_code:
                args[field_name] = destination_code

        elif name in {
            "dep",
            "departure",
            "origin",
            "from",
            "source",
        }:
            if origin_code:
                args[field_name] = origin_code

        elif name in {
            "arr",
            "arrival",
            "destination",
            "dest",
            "to",
        }:
            if destination_code:
                args[field_name] = destination_code

    return args


def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n", flush=True)

    query = state["user_query"]
    constraints = state.get("trip_constraints", {})

    airports = None
    airlines = None
    live_flight_data = None
    live_search_error = None

    try:
        # Keep the existing AviationStack reference calls.
        airports = asyncio.run(
            aviation_mcp_call("list_airports")
        )

        airlines = asyncio.run(
            aviation_mcp_call("list_airlines")
        )

        print("\nAIRPORTS:", airports, flush=True)
        print("\nAIRLINES:", airlines, flush=True)

        if _mcp_payload_has_error(airports):
            raise RuntimeError(
                "AviationStack airport reference request returned an API error."
            )

        if _mcp_payload_has_error(airlines):
            raise RuntimeError(
                "AviationStack airline reference request returned an API error."
            )

        # Resolve the origin/destination from supervisor constraints.
        origin = str(
            constraints.get("origin")
            or ""
        ).strip()

        destination = str(
            constraints.get("destination")
            or ""
        ).strip()

        # If the supervisor did not extract them, use the user's query as the
        # source of truth and let the LLM extract the two locations.
        if not origin or not destination:
            extraction_prompt = f"""
Extract the origin and destination from this travel request.

Return strict JSON only:
{{
  "origin": "",
  "destination": ""
}}

Travel request:
{query}
"""

            extracted = _json_from_llm(
                _llm_text(
                    "Extract travel route information. Return JSON only.",
                    extraction_prompt,
                )
            )

            origin = str(
                extracted.get("origin") or origin
            ).strip()

            destination = str(
                extracted.get("destination") or destination
            ).strip()

        origin_code = _airport_code_from_reference(
            airports,
            origin,
        )

        destination_code = _airport_code_from_reference(
            airports,
            destination,
        )

        print(
            f"FLIGHT ROUTE: {origin} ({origin_code}) -> "
            f"{destination} ({destination_code})",
            flush=True,
        )

        # Discover the real flight-search tool exposed by aviationstack-mcp.
        flight_tool = asyncio.run(
            _find_aviation_flight_tool()
        )

        if flight_tool is None:
            raise RuntimeError(
                "No compatible live route-search tool was exposed by "
                "aviationstack-mcp. Available AviationStack reference tools "
                "are not sufficient for a live origin-to-destination search."
            )

        flight_tool_args = _build_flight_tool_args(
            flight_tool,
            origin_code,
            destination_code,
        )

        print(
            "AVIATION FLIGHT TOOL:",
            flight_tool.name,
            flush=True,
        )

        print(
            "AVIATION FLIGHT TOOL ARGS:",
            flight_tool_args,
            flush=True,
        )

        # If the discovered tool requires route arguments and we couldn't map
        # them, don't fabricate a call. Use the generic fallback instead.
        if not flight_tool_args:
            raise RuntimeError(
                f"Could not map origin/destination to the schema of "
                f"AviationStack tool '{flight_tool.name}'."
            )

        live_flight_data = asyncio.run(
            flight_tool.ainvoke(flight_tool_args)
        )

        print(
            "\nLIVE FLIGHT DATA:",
            live_flight_data,
            flush=True,
        )

        if _mcp_payload_has_error(live_flight_data):
            raise RuntimeError(
                "AviationStack flight search returned an API error."
            )

    except Exception as exc:
        live_search_error = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "FLIGHT SEARCH FALLBACK:",
            live_search_error,
            flush=True,
        )

    # ---------------------------------------------------------
    # LLM response generation
    # ---------------------------------------------------------
    if live_search_error:
        flight_data_for_llm = f"""
AviationStack live flight search is unavailable.

Reason:
{live_search_error}

Airport reference data:
{str(airports)[:3000] if airports else "Unavailable"}

Airline reference data:
{str(airlines)[:3000] if airlines else "Unavailable"}

Use general travel knowledge for the response, but clearly label
flight duration, airline suggestions, and airfare as estimates.
Do not invent live flight numbers, live schedules, or live statuses.
"""
    else:
        flight_data_for_llm = f"""
LIVE AVIATIONSTACK FLIGHT SEARCH RESULT:
{str(live_flight_data)[:6000]}

AIRPORT REFERENCE DATA:
{str(airports)[:2000]}

AIRLINE REFERENCE DATA:
{str(airlines)[:2000]}
"""

    try:
        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            flight_data=flight_data_for_llm,
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an expert travel flight planner."
                ),
                HumanMessage(content=prompt),
            ]
        )

        flight_data = str(response.content)

        if live_search_error:
            flight_data = (
                "Live AviationStack flight search was unavailable. "
                "The following is general/estimated flight guidance, "
                "not live flight data.\n\n"
                f"{flight_data}"
            )

    except Exception as exc:
        print(
            "FLIGHT AGENT LLM ERROR:",
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        # Preserve the requested generic fallback behavior.
        flight_data = (
            "Live flight information is temporarily unavailable. "
            "General flight-planning guidance can be provided, but "
            "current schedules and prices should be verified with "
            "the airline or booking provider."
        )

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Hotel Agent - original behavior kept
# =========================
def hotel_agent(state: TravelState):
    query = (
        f"Best hotels for "
        f"{state['user_query']}"
    )

    try:
        hotel_results = asyncio.run(
            tavily_mcp_search(query)
        )

    except Exception as exc:
        print(
            f"HOTEL AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + 1
        ),
    }


# =========================
# Weather Agent - original behavior kept
# =========================
def weather_agent(state: TravelState):
    try:
        city = extract_destination(
            state["user_query"]
        )

        weather_data = asyncio.run(
            weather_mcp_search(city)
        )

        forecast_data = asyncio.run(
            forecast_mcp_search(city)
        )

        weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""

    except Exception as exc:
        print(
            f"WEATHER AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        weather_results = (
            "Live weather information "
            "is temporarily unavailable. Give general "
            "seasonal guidance and advise the traveler "
            "to verify the forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
    }


# =========================
# Budget Agent - new specialist
# =========================
def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent - original behavior extended with selected results
# =========================
def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Budget Results:
{state.get('budget_results', '')}

Make the itinerary practical, budget-aware, and easy to follow.
Create a clear draft that is ready for human review.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent - original format kept, HITL feedback added
# =========================
def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights:
{state.get('flight_results', '')}

Hotels:
{state.get('hotel_results', '')}

Weather:
{state.get('weather_results', '')}

Budget Analysis:
{state.get('budget_results', '')}

Draft Itinerary:
{state.get('itinerary', '')}

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
- Incorporate the human feedback when revision was requested.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a professional AI travel booking assistant."
            ),
            HumanMessage(content=final_prompt),
        ]
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

# =========================
# PostgreSQL Checkpointer - original persistence kept
# =========================
DATABASE_URL = get_database_url()
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)


def stream_run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and yield intermediate nodes in real-time."""
    import traceback
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    yield {
        "type": "start",
        "thread_id": thread_id,
        "message": f"Initializing LangGraph multi-agent flow (Thread: {thread_id})..."
    }

    inputs = {
        "messages": [HumanMessage(content=user_input)],
        "user_query": user_input,
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "selected_agents": [],
        "trip_constraints": _empty_constraints(),
        "supervisor_reasoning": "",
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "budget_results": "",
        "itinerary": "",
        "approval_request": "",
        "approved": False,
        "human_feedback": "",
        "final_response": "",
        "llm_calls": 0,
    }

    try:
        for chunk in travel_graph.stream(
            inputs,
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_data in chunk.items():
                # Human review pauses the graph with an interrupt tuple. The
                # pause is handled from the persisted graph state below, so it
                # must not be treated as a normal node-update dictionary.
                if node_name == "__interrupt__" or not isinstance(node_data, dict):
                    continue

                message = f"Node '{node_name}' completed execution."
                if node_name == "supervisor":
                    message = "Supervisor reviewed query constraints and completed routing."
                elif node_name == "flight_agent":
                    message = "Flight Agent processed AviationStack flight data with LLM fallback when live search was unavailable."
                elif node_name == "hotel_agent":
                    message = "Hotel Agent finished neighborhood and lodging analysis."
                elif node_name == "weather_agent":
                    message = "Weather Agent completed seasonal forecast evaluations."
                elif node_name == "budget_agent":
                    message = "Budget Agent calculated pricing models and cost feasibility."
                elif node_name == "itinerary_agent":
                    message = "Itinerary Agent prepared the draft planner schedule."
                elif node_name == "final_agent":
                    message = "Final Polisher Agent synthesized and optimized the itinerary."

                yield {
                    "type": "node_complete",
                    "node": node_name,
                    "message": message,
                    "data": {
                        "selected_agents": node_data.get("selected_agents"),
                        "supervisor_reasoning": node_data.get("supervisor_reasoning"),
                        "guardrail_allowed": node_data.get("guardrail_allowed"),
                        "guardrail_reason": node_data.get("guardrail_reason"),
                        "flight_results": node_data.get("flight_results"),
                        "hotel_results": node_data.get("hotel_results"),
                        "weather_results": node_data.get("weather_results"),
                        "budget_results": node_data.get("budget_results"),
                        "itinerary": node_data.get("itinerary"),
                        "final_response": node_data.get("final_response"),
                        "approval_request": node_data.get("approval_request"),
                    }
                }

        # Check for pause state after stream iteration
        state = travel_graph.get_state(config)
        if state.next and state.next[0] == "human_approval":
            yield {
                "type": "interrupt",
                "message": "Human verification requested. Standing by for feedback...",
                "data": {
                    "thread_id": thread_id,
                    "requires_approval": True,
                    "approval_request": state.values.get("approval_request") or "Please review the draft plan.",
                    "itinerary": state.values.get("itinerary") or state.values.get("final_response"),
                    "flight_results": state.values.get("flight_results"),
                    "hotel_results": state.values.get("hotel_results"),
                    "weather_results": state.values.get("weather_results"),
                    "budget_results": state.values.get("budget_results"),
                    "selected_agents": state.values.get("selected_agents"),
                    "supervisor_reasoning": state.values.get("supervisor_reasoning"),
                    "trip_constraints": state.values.get("trip_constraints"),
                }
            }
        else:
            yield {
                "type": "complete",
                "message": "Workflow successfully finished.",
                "data": _serialize_result(state.values, thread_id)
            }

    except Exception as exc:
        traceback.print_exc()
        yield {
            "type": "error",
            "message": f"Error running travel agent: {exc}"
        }


def stream_resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread and yield updates in real-time."""
    import traceback
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}

    yield {
        "type": "resume",
        "thread_id": thread_id,
        "message": f"Resuming agent thread with approved={approved}..."
    }

    try:
        for chunk in travel_graph.stream(
            Command(
                resume={
                    "approved": approved,
                    "feedback": feedback.strip(),
                }
            ),
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_data in chunk.items():
                message = f"Node '{node_name}' completed execution."
                if node_name == "human_approval":
                    message = "Human review processed successfully."
                elif node_name == "final_agent":
                    message = "Final Polisher Agent completed the travel itinerary details."

                yield {
                    "type": "node_complete",
                    "node": node_name,
                    "message": message,
                    "data": {
                        "approved": node_data.get("approved"),
                        "human_feedback": node_data.get("human_feedback"),
                        "final_response": node_data.get("final_response"),
                    }
                }

        state = travel_graph.get_state(config)
        yield {
            "type": "complete",
            "message": "Workflow successfully resumed and finished.",
            "data": _serialize_result(state.values, thread_id)
        }

    except Exception as exc:
        traceback.print_exc()
        yield {
            "type": "error",
            "message": f"Error resuming travel agent: {exc}"
        }