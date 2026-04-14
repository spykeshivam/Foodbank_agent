"""Foodbank Admin AI Chat Interface — Streamlit app."""
import pandas as pd
import plotly.io as pio
import streamlit as st

from agent import AgentResponse, run_query
from sheets import fetch_logins, fetch_registrations
from tools import init_datasets

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Foodbank Admin AI",
    page_icon="🥫",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    registrations = pd.DataFrame(fetch_registrations())
    logins = pd.DataFrame(fetch_logins())
    return registrations, logins


registrations_df, logins_df = load_data()
init_datasets(registrations_df, logins_df)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # display history [{role, content}]
if "api_history" not in st.session_state:
    st.session_state.api_history = []       # Gemini Content objects
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None   # clarification question str


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_message(role: str, content) -> None:
    avatar = "👤" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        if isinstance(content, str):
            st.markdown(content)
        elif isinstance(content, list):
            for block in content:
                if block["type"] == "text":
                    st.markdown(block["text"])
                elif block["type"] == "dataframe":
                    st.dataframe(block["data"], use_container_width=True)
                elif block["type"] == "chart":
                    try:
                        fig = pio.from_json(open(block["path"]).read())
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not render chart: {e}")


def handle_response(result: AgentResponse) -> None:
    if result.clarification_question:
        st.session_state.pending_clarification = result.clarification_question
        st.session_state.messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": result.clarification_question}],
        })
    else:
        st.session_state.pending_clarification = None
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.display_blocks,
        })


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🥫 Foodbank Admin AI")
st.caption("Ask questions about registrations and visit data in plain English.")

with st.sidebar:
    st.header("Dataset Info")
    st.metric("Registered Users", len(registrations_df))
    st.metric("Total Logins", len(logins_df))
    st.divider()
    st.subheader("Example Questions")
    examples = [
        "How many people have Halal dietary requirements?",
        "Show me the gender breakdown as a pie chart.",
        "How many logins were there each day?",
        "What are the top 5 spoken languages?",
        "How many female users with Halal requirements visited in the past few months?",
    ]
    for ex in examples:
        st.markdown(f"- *{ex}*")
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.api_history = []
        st.session_state.pending_clarification = None
        st.rerun()

for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

placeholder = (
    "Answer the clarification above…"
    if st.session_state.pending_clarification
    else "Ask a question about your foodbank data…"
)

user_input = st.chat_input(placeholder)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    render_message("user", user_input)

    with st.spinner("Thinking…"):
        result = run_query(user_input, history=st.session_state.api_history)
        handle_response(result)

    st.rerun()
