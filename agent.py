# %% Imports & configuration
import os
import sqlite3
from typing import Annotated, Literal

import requests
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

try:  # optional: read GROQ_API_KEY from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DB_PATH = "tasks.db"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
HEADERS = {"User-Agent": "StudentAssistantAgent/1.0 (course project)"}


def _db():
    """Open the SQLite task database (creates the table if needed)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tasks "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL, task TEXT NOT NULL)"
    )
    return conn


def _owner(config: RunnableConfig) -> str:
    """Each conversation (thread_id) has its own private task list."""
    return config.get("configurable", {}).get("thread_id", "default")


def get_tasks(owner: str):
    """Returns [(id, task), ...] for one conversation/user."""
    with _db() as conn:
        return conn.execute(
            "SELECT id, task FROM tasks WHERE owner = ? ORDER BY id", (owner,)
        ).fetchall()


# %% Tools
@tool
def add_task(task: str, config: RunnableConfig) -> str:
    """
    Adds a new task to the user's task list and returns its ID.
    """
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (owner, task) VALUES (?, ?)", (_owner(config), task)
        )
        return f"Task added: #{cur.lastrowid} - {task}"


@tool
def list_tasks(config: RunnableConfig) -> str:
    """
    Returns all tasks in the user's task list together with their IDs.
    """
    rows = get_tasks(_owner(config))
    if not rows:
        return "The task list is empty."
    return "\n".join(f"#{i}: {t}" for i, t in rows)


@tool
def delete_task(task_id: int, config: RunnableConfig) -> str:
    """
    Permanently deletes the task with the given ID. This is a DESTRUCTIVE
    action - the graph asks the user for approval before it runs.
    """
    with _db() as conn:
        cur = conn.execute(
            "DELETE FROM tasks WHERE id = ? AND owner = ?", (task_id, _owner(config))
        )
        if cur.rowcount == 0:
            return f"No task with ID #{task_id} was found."
        return f"Task #{task_id} deleted."


@tool
def get_weather(city: str) -> str:
    """
    Gets the current weather (temperature, wind, humidity) for a city using
    the free Open-Meteo API (no API key needed).
    """
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            headers=HEADERS,
            timeout=20,
        ).json()
        if not geo.get("results"):
            return f"City '{city}' was not found."
        place = geo["results"][0]
        data = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,wind_speed_10m,relative_humidity_2m,precipitation",
            },
            headers=HEADERS,
            timeout=20,
        ).json()["current"]
        return (
            f"Weather in {place['name']}, {place.get('country', '')}: "
            f"{data['temperature_2m']}°C, wind {data['wind_speed_10m']} km/h, "
            f"humidity {data['relative_humidity_2m']}%, precipitation {data['precipitation']} mm."
        )
    except Exception as e:
        return f"Could not get the weather: {e}"


@tool
def search_books(topic: str) -> str:
    """
    Searches books by topic using the free Open Library Search API and
    returns up to 5 titles with authors.
    """
    try:
        r = requests.get(
            "https://openlibrary.org/search.json",
            params={"q": topic, "limit": 5, "fields": "title,author_name,first_publish_year"},
            headers=HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        docs = r.json().get("docs", [])
        if not docs:
            return f"No books found for '{topic}'."
        return "\n".join(
            f"- {d.get('title', '?')} — {', '.join(d.get('author_name', ['unknown'])[:2])}"
            f" ({d.get('first_publish_year', 'n/a')})"
            for d in docs
        )
    except Exception as e:
        return f"Book search failed: {e}"


TOOLS = [add_task, list_tasks, delete_task, get_weather, search_books]
DESTRUCTIVE_TOOLS = {"delete_task"}


# %% State
class State(TypedDict):
    """Conversation state: the list of messages (appended, never overwritten)."""

    messages: Annotated[list, add_messages]


# %% Graph
SYSTEM_PROMPT = (
    "You are a practical student assistant. You manage a task list, check the "
    "weather (Open-Meteo) and search books (Open Library). Reply in the same "
    "language the user writes in. If you need a task ID, call list_tasks first. "
    "When the user asks to delete something, call delete_task - the system "
    "itself asks the user for confirmation, so do not ask twice. If a tool "
    "result says the action was cancelled, tell the user nothing was changed. "
    "Personal touch: after giving the weather, suggest a place to study "
    "(outdoors/park if it is warm and dry, library or cafe if it is cold or "
    "rainy). Never invent task IDs, temperatures or book titles."
)
YES = {"true", "yes", "y", "ok", "კი", "დიახ", "დაადასტურე", "დადასტურება"}


def is_yes(answer) -> bool:
    """True for resume=True or an affirmative text answer."""
    return answer is True or str(answer).strip().lower() in YES


def build_graph(checkpointer=None):
    """Builds and compiles the agent graph."""
    llm = ChatGroq(model=MODEL, temperature=0, max_tokens=800).bind_tools(TOOLS)

    def assistant(state: State):
        reply = llm.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [reply]}

    def route(state: State) -> Literal["tools", "approval", "__end__"]:
        calls = state["messages"][-1].tool_calls
        if not calls:
            return END
        if any(c["name"] in DESTRUCTIVE_TOOLS for c in calls):
            return "approval"
        return "tools"

    def approval(state: State, config: RunnableConfig) -> Command[Literal["tools", "cancel"]]:
        calls = state["messages"][-1].tool_calls
        owner = _owner(config)
        descriptions = []
        for c in calls:
            if c["name"] in DESTRUCTIVE_TOOLS:
                tid = c["args"].get("task_id")
                text = dict(get_tasks(owner)).get(tid, "unknown task")
                descriptions.append(f"delete task #{tid} (“{text}”)")
        answer = interrupt(  # graph pauses here until the user answers
            {
                "message": "⚠️ Confirm: " + "; ".join(descriptions) + ". Proceed?",
                "actions": [c for c in calls if c["name"] in DESTRUCTIVE_TOOLS],
            }
        )
        return Command(goto="tools" if is_yes(answer) else "cancel")

    def cancel(state: State):
        return {
            "messages": [
                ToolMessage(
                    "CANCELLED: the user denied this action. Nothing was changed.",
                    tool_call_id=c["id"],
                )
                for c in state["messages"][-1].tool_calls
            ]
        }

    g = StateGraph(State)
    g.add_node("assistant", assistant)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("approval", approval)
    g.add_node("cancel", cancel)
    g.add_edge(START, "assistant")
    g.add_conditional_edges("assistant", route)
    g.add_edge("tools", "assistant")
    g.add_edge("cancel", "assistant")
    return g.compile(checkpointer=checkpointer or InMemorySaver())


# %% Helpers
def pending_approval(graph, config):
    """Returns the approval request dict if the graph is paused, else None."""
    for task in graph.get_state(config).tasks:
        for item in task.interrupts:
            return item.value
    return None


def last_text(result) -> str:
    """Text of the last message in a graph result."""
    return result["messages"][-1].content


def friendly_error(e: Exception) -> str:
    """Readable message for API problems (missing key, rate limit, network)."""
    s = str(e).lower()
    if "rate" in s or "429" in s:
        return "⏳ The AI service rate limit was reached. Please wait a bit and try again."
    if "api key" in s or "401" in s or "authentication" in s:
        return "🔑 API key problem. Check GROQ_API_KEY."
    return f"⚠️ The service is not responding right now: {e}"
