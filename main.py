"""Terminal chat with the agent. Run: python main.py"""
import sqlite3
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent import build_graph, friendly_error, is_yes, last_text, pending_approval

# SQLite checkpointer: conversation memory survives restarts.
conn = sqlite3.connect("memory.db", check_same_thread=False)
graph = build_graph(SqliteSaver(conn))
config = {"configurable": {"thread_id": "cli-user"}}

print("Student Assistant Agent. Type 'exit' to quit, '/new' for a fresh conversation.")
while True:
    text = input("\nYou: ").strip()
    if text.lower() in {"exit", "quit"}:
        break
    if text == "/new":
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        print("Started a new conversation.")
        continue
    if not text:
        continue
    try:
        result = graph.invoke({"messages": [("user", text)]}, config)
        while (request := pending_approval(graph, config)) is not None:
            answer = input(f"\n{request['message']} (yes/no)\nYou: ")
            result = graph.invoke(Command(resume=is_yes(answer)), config)
        print(f"\nAgent: {last_text(result)}")
    except Exception as e:
        print(f"\n{friendly_error(e)}")
