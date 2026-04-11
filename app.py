"""Foodbank Admin AI Chat Interface — Streamlit app (google-genai SDK)."""
import json
import os

import pandas as pd
import plotly.io as pio
import streamlit as st
from google import genai
from google.genai import types

from tool_schemas import TOOL_SCHEMAS
from tools import TOOL_FUNCTIONS, init_datasets

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Foodbank Admin AI",
    page_icon="🥫",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    registrations = pd.read_excel(
        os.path.join(DATA_DIR, "Registration Form (Responses).xlsx")
    )
    logins = pd.read_excel(
        os.path.join(DATA_DIR, "Log In (Responses).xlsx")
    )
    return registrations, logins


registrations_df, logins_df = load_data()
init_datasets(registrations_df, logins_df)

# ── Gemini client & model ─────────────────────────────────────────────────────
_gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_PROMPT = """You are a data analyst assistant for a foodbank called St Dunstan's Food Bank.
You have access to two datasets:
- registrations (one row per registered user) — key columns include:
  Username, First Name, Surname, Date of Birth, Sex, Postcode, Primary Spoken Language,
  Dietary Requirements, Ethnicity, Relationship Status, Property Type,
  Number of Adults in Household, Number of Children in Household,
  Number of children in each age range [Under 5 / 5-11 / 11-16 / 16-18],
  Number of people on Benefits [Universal credit / Benefits / Unemployed / Retired /
  Minimum wage / Over £25K / Under £25K / Pension], Cooking Facilities, Timestamp.
- logins (one row per visit) — columns: Username, Timestamp, Day.

The two datasets join on Username.

Always use tools to retrieve real data — never guess or invent numbers.
If a question is ambiguous (e.g. a time period is not specified), use clarify_question first.
Every response must include at least one numerical result and, where meaningful, at least one chart.
When creating charts, pass the 'data' array from group_and_count serialised as a JSON string.
Be concise and empathetic in tone — remember these results reflect real people in need."""

# Build the Gemini Tool object once at startup
_gemini_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name=s["name"],
            description=s["description"],
            parameters_json_schema=s["parameters"],
        )
        for s in TOOL_SCHEMAS
    ]
)

_generate_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[_gemini_tool],
    temperature=0.2,
)

MAX_TOOL_CALLS = 10

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # display history [{role, content}]
if "api_messages" not in st.session_state:
    st.session_state.api_messages = []      # Gemini Content history
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None   # {function_call_id, question}
if "pending_tool_responses" not in st.session_state:
    st.session_state.pending_tool_responses = []    # tool responses before clarification


# ── Helpers ───────────────────────────────────────────────────────────────────

def _args_to_dict(args) -> dict:
    """Convert Gemini MapComposite args to a plain Python dict."""
    if args is None:
        return {}
    result = {}
    for k, v in args.items():
        if hasattr(v, "items"):          # nested MapComposite → dict
            result[k] = dict(v)
        elif hasattr(v, "__iter__") and not isinstance(v, str):  # list
            result[k] = list(v)
        else:
            result[k] = v
    return result


def call_tool(tool_name: str, tool_input: dict) -> str:
    fn = TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return fn(**tool_input)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


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


def _finalize_response(text_blocks: list[str], collected_results: list[dict]) -> None:
    display_blocks = []

    if text_blocks:
        display_blocks.append({"type": "text", "text": "\n\n".join(text_blocks)})

    for item in collected_results:
        if item["name"] == "group_and_count" and "data" in item["result"]:
            df = pd.DataFrame(item["result"]["data"])
            display_blocks.append({"type": "dataframe", "data": df})

    for item in collected_results:
        if "chart_path" in item["result"]:
            display_blocks.append({"type": "chart", "path": item["result"]["chart_path"]})

    st.session_state.messages.append({"role": "assistant", "content": display_blocks})


# ── Core agentic loop ─────────────────────────────────────────────────────────

def _run_loop(tool_call_count: int = 0, collected_results: list | None = None) -> None:
    """
    Inner loop — shared by first call and post-clarification continuation.
    Mutates st.session_state.api_messages throughout.
    """
    if collected_results is None:
        collected_results = []

    while tool_call_count < MAX_TOOL_CALLS:
        response = _gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=st.session_state.api_messages,
            config=_generate_config,
        )

        # Parse response parts
        text_parts: list[str] = []
        function_calls: list = []

        candidate = response.candidates[0]
        for part in candidate.content.parts:
            if part.text:
                text_parts.append(part.text)
            if part.function_call and part.function_call.name:
                function_calls.append(part.function_call)

        # Append model turn to history
        st.session_state.api_messages.append(candidate.content)

        # No more tool calls — produce final answer
        if not function_calls:
            _finalize_response(text_parts, collected_results)
            return

        # Process each function call
        tool_responses: list[types.Part] = []
        clarification_intercepted = False

        for fc in function_calls:
            tool_call_count += 1
            args = _args_to_dict(fc.args)
            result_str = call_tool(fc.name, args)
            result_data = json.loads(result_str)

            if fc.name == "clarify_question" and result_data.get("clarification_needed"):
                # Pause loop — save pending state
                st.session_state.pending_clarification = {
                    "fc_name": fc.name,
                    "question": result_data["question"],
                    "tool_call_count": tool_call_count,
                    "collected_results": collected_results,
                }
                # Save any tool responses we built before hitting clarification
                st.session_state.pending_tool_responses = tool_responses[:]
                # Surface the question in chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": result_data["question"]}],
                })
                clarification_intercepted = True
                break

            tool_responses.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result_str},
                )
            )
            collected_results.append({"name": fc.name, "result": result_data})

        if clarification_intercepted:
            return

        # Feed tool results back as a user turn
        st.session_state.api_messages.append(
            types.Content(role="user", parts=tool_responses)
        )

    # Exceeded max tool calls — ask for final answer
    st.session_state.api_messages.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text("Please provide a final answer now based on the data you have retrieved.")],
        )
    )
    response = _gemini_client.models.generate_content(
        model="gemini-1.5-flash",
        contents=st.session_state.api_messages,
        config=_generate_config,
    )
    texts = [p.text for p in response.candidates[0].content.parts if p.text]
    _finalize_response(texts, collected_results)


def run_agentic_loop(user_text: str) -> None:
    st.session_state.api_messages.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(user_text)],
        )
    )
    _run_loop()


def resume_after_clarification(user_answer: str) -> None:
    pending = st.session_state.pending_clarification
    st.session_state.pending_clarification = None

    # Build the function response for the clarify_question call
    clarify_response = types.Part.from_function_response(
        name=pending["fc_name"],
        response={"user_answer": user_answer},
    )

    # Combine any earlier tool responses with the clarification response
    all_responses = st.session_state.pending_tool_responses + [clarify_response]
    st.session_state.pending_tool_responses = []

    st.session_state.api_messages.append(
        types.Content(role="user", parts=all_responses)
    )

    _run_loop(
        tool_call_count=pending["tool_call_count"],
        collected_results=pending["collected_results"],
    )


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
        st.session_state.api_messages = []
        st.session_state.pending_clarification = None
        st.session_state.pending_tool_responses = []
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
        if st.session_state.pending_clarification:
            resume_after_clarification(user_input)
        else:
            run_agentic_loop(user_input)

    st.rerun()
