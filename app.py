"""Streamlit chat interface. Run: streamlit run app.py"""
import os
import uuid

import streamlit as st
from langgraph.types import Command

try:  # Streamlit Cloud: key comes from Secrets, never from GitHub
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # local run: use .env / environment variable

from agent import build_graph, friendly_error, get_tasks, last_text, pending_approval

st.set_page_config(page_title="Student Assistant Agent", page_icon="🎓")
st.title("🎓 Student Assistant Agent")
st.caption("Manage your tasks, check the weather and find books.")


@st.cache_resource
def get_graph():
    return build_graph()  # built once, shared; memory is separated by thread_id


graph = get_graph()

if "thread_id" not in st.session_state:  # one private conversation per user session
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []
thread_id = st.session_state.thread_id
config = {"configurable": {"thread_id": thread_id}}


def handle(payload):
    try:
        result = graph.invoke(payload, config)
        if pending_approval(graph, config) is None:
            st.session_state.history.append(("assistant", last_text(result)))
    except Exception as e:
        st.session_state.history.append(("assistant", friendly_error(e)))
    st.rerun()


with st.sidebar:
    st.header("📝 My tasks")
    rows = get_tasks(thread_id)
    st.write("\n\n".join(f"**#{i}** {t}" for i, t in rows) or "_empty_")
    if st.button("New conversation"):
        st.session_state.clear()
        st.rerun()

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

request = pending_approval(graph, config)
if request:
    with st.chat_message("assistant"):
        st.warning(request["message"])
        col1, col2 = st.columns(2)
        if col1.button("✅ Confirm"):
            handle(Command(resume=True))
        if col2.button("❌ Cancel"):
            handle(Command(resume=False))
elif prompt := st.chat_input("Ask me anything..."):
    st.session_state.history.append(("user", prompt))
    handle({"messages": [("user", prompt)]})
