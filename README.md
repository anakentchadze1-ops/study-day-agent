# Student Assistant Agent

This agent helps students manage a task list, check the weather and search books
using LangGraph, a Groq LLM and two external APIs (Open-Meteo, Open Library).
A StateGraph routes between the assistant, safe tools and a human-approval step,
so deleting a task only happens after the user confirms. After giving the weather
it suggests where to study (outdoors or library/cafe).

## Features
- **Tools:** `add_task`, `list_tasks`, `delete_task` (needs approval), `get_weather` (Open-Meteo), `search_books` (Open Library)
- **Memory:** checkpointer + `thread_id` (InMemorySaver in notebook/Streamlit, SQLite in CLI)
- **Approval flow:** `interrupt()` pauses the graph; `Command(resume=True)` deletes, `Command(resume=False)` cancels
- **Per-user tasks:** each conversation (`thread_id`) has its own task list
- **Interfaces:** Jupyter/Colab notebook, Streamlit chat, terminal chat

## Setup
```bash
pip install langgraph langchain-groq requests streamlit python-dotenv langgraph-checkpoint-sqlite
# or: pip install -r requirements.txt
```
Create `.env` (see `.env.example`) and add your own Groq key from https://console.groq.com/keys :
```
GROQ_API_KEY=""
```
Never commit a real key (`.env` and `.streamlit/secrets.toml` are in `.gitignore`).
On Streamlit Cloud add the key under *Advanced settings → Secrets*:
`GROQ_API_KEY = "your_key"`.

## Run
```bash
python main.py                                   # terminal chat
streamlit run app.py                             # web chat
jupyter notebook practical_assistant_agent.ipynb # tests + evaluation
```

## Files
| File | Purpose |
|------|---------|
| `agent.py` | tools, State, StateGraph, approval logic (`build_graph()`) |
| `main.py` | CLI version |
| `app.py` | Streamlit chat |
| `practical_assistant_agent.ipynb` | diagram, 6 tests, approve/deny demos, evaluation table |

## Graph
`START → assistant → (tools | approval | END)`; `approval → tools` if confirmed, `approval → cancel` if denied; `tools / cancel → assistant`.

## Limitations
Free Groq plan has rate limits. Streamlit/notebook memory is lost on restart (the CLI keeps it in SQLite).
