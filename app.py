"""Foodbank Admin AI Chat Interface — Streamlit app."""
import hmac
import os

import pandas as pd
import plotly.io as pio
import streamlit as st

from agent import AgentResponse, continue_after_clarification, run_query
from log_config import get_logger, setup_logging
from sheets import fetch_logins, fetch_registrations
from tools import init_datasets

setup_logging()
log = get_logger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Foodbank Admin AI",
    page_icon="🥫",
    layout="wide",
)

# ── Auth gate ─────────────────────────────────────────────────────────────────
_APP_USER = os.environ["APP_USERNAME"]
_APP_PASS = os.environ["APP_PASSWORD"]


def _login_gate() -> None:
    if st.session_state.get("authenticated"):
        return
    st.title("🥫 Foodbank Admin AI")
    st.subheader("Sign in")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        user_ok = hmac.compare_digest(username, _APP_USER)
        pass_ok = hmac.compare_digest(password, _APP_PASS)
        if user_ok and pass_ok:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()


_login_gate()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    registrations = pd.DataFrame(fetch_registrations())
    logins = pd.DataFrame(fetch_logins())
    return registrations, logins


registrations_df, logins_df = load_data()
init_datasets(registrations_df, logins_df)
log.info("App started — registrations: %d rows, logins: %d rows", len(registrations_df), len(logins_df))

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # display history [{role, content}]
if "api_history" not in st.session_state:
    st.session_state.api_history = []       # Gemini Content objects
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None   # paused AgentResponse or None


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
    st.session_state.api_history = result.history   # persist multi-turn context
    if result.clarification_question:
        st.session_state.pending_clarification = result   # store full paused state
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

    _retry_msg = st.empty()

    def _on_retry():
        log.warning("Server busy — showing retry banner to user")
        _retry_msg.warning(
            "Our servers are experiencing high demand — please wait, retrying…"
        )

    try:
        with st.spinner("Thinking…"):
            if st.session_state.pending_clarification is not None:
                log.info("User answered clarification: %r", user_input[:120])
                result = continue_after_clarification(
                    user_input, st.session_state.pending_clarification, on_retry=_on_retry
                )
            else:
                log.info("New user query: %r", user_input[:120])
                result = run_query(
                    user_input, history=st.session_state.api_history, on_retry=_on_retry
                )

        _retry_msg.empty()
        log.info(
            "Response ready — clarification=%s, display_blocks=%d, tools=%s",
            result.clarification_question is not None,
            len(result.display_blocks),
            result.tool_calls,
        )
        handle_response(result)

    except Exception as exc:
        _retry_msg.empty()
        error_name = type(exc).__name__
        # First line of the message only — strip internal traceback noise
        error_detail = str(exc).splitlines()[0]
        log.error("Unhandled error during query: %s", exc, exc_info=True)
        with st.chat_message("assistant", avatar="🤖"):
            st.error(
                f"**{error_name}:** {error_detail}\n\n"
                "Contact support to get the issue resolved."
            )
        st.session_state.messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": f"⚠️ {error_name}: {error_detail}"}],
        })

    st.rerun()
