"""
Core agentic loop — no Streamlit dependency.

app.py and tests both import from here.
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import httpx
import openai
import pandas as pd

from log_config import get_logger, setup_logging
from tool_schemas import TOOL_SCHEMAS
from tools import TOOL_FUNCTIONS

setup_logging()
log = get_logger(__name__)

MAX_TOOL_CALLS = 10

# Sliding window of API call timestamps for rate tracking
_api_call_times: deque[float] = deque()
_RATE_WINDOW = 300  # 5 minutes in seconds

PROVIDERS: dict[str, dict] = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "Cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "model": "llama-3.3-70b",
    },
}

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


def _record_api_call() -> None:
    now = time.monotonic()
    _api_call_times.append(now)
    cutoff = now - _RATE_WINDOW
    while _api_call_times and _api_call_times[0] < cutoff:
        _api_call_times.popleft()
    log.info("API requests in last 5 min: %d", len(_api_call_times))


@dataclass
class AgentResponse:
    """Structured result returned by run_query() and continue_after_clarification()."""

    text: str  # final natural-language answer
    tool_calls: list[str]  # ordered list of tool names called
    display_blocks: list[dict]  # [{type, ...}] ready for rendering
    clarification_question: str | None  # non-None if agent paused to clarify
    history: list = field(default_factory=list)  # conversation turns (no system msg) — persist in session

    # ── Internal resume state (populated only when clarification_question is set) ──
    _paused_messages: list = field(default_factory=list)
    _paused_tool_call_id: str = ""  # OpenAI tool_call id of the clarify_question call
    _paused_collected: list = field(default_factory=list)
    _paused_tool_call_count: int = 0


def _build_client(provider: str) -> tuple[openai.OpenAI, str]:
    cfg = PROVIDERS[provider]
    client = openai.OpenAI(
        base_url=cfg["base_url"],
        api_key=os.environ[cfg["api_key_env"]],
    )
    return client, cfg["model"]


def _build_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        }
        for s in TOOL_SCHEMAS
    ]


def _build_system_message() -> dict:
    today = datetime.now().strftime("%A, %d %B %Y")
    content = (
        f"The current date is {today}. "
        "This is injected at runtime and is always accurate — "
        "do not rely on your training cutoff to infer what 'now', 'recent', or 'this year' means.\n\n" + SYSTEM_PROMPT
    )
    return {"role": "system", "content": content}


def _generate_with_retry(client, model, messages, tools, max_retries=5, on_retry=None):
    """Call chat.completions.create with linear 40s backoff on 429 and fixed retry on 5xx."""
    rate_limit_attempts = 0
    attempt_total = 0
    while True:
        attempt_total += 1
        log.debug("API call attempt %d (model=%s, messages=%d)", attempt_total, model, len(messages))
        try:
            _record_api_call()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                temperature=0.2,
                timeout=600,
            )
            u = response.usage
            log.info(
                "Tokens — prompt: %d, output: %d, total: %d",
                u.prompt_tokens or 0,
                u.completion_tokens or 0,
                u.total_tokens or 0,
            )
            return response
        except openai.RateLimitError:
            rate_limit_attempts += 1
            log.warning("429 rate-limit hit (attempt %d/%d) — waiting 40s", rate_limit_attempts, max_retries)
            if rate_limit_attempts >= max_retries:
                log.error("429 rate-limit: max retries exceeded")
                raise
            time.sleep(40)
        except openai.APIStatusError as e:
            if e.status_code in (503, 504):
                rate_limit_attempts += 1
                log.warning(
                    "APIStatusError %d (attempt %d/%d) — retrying in 5s",
                    e.status_code,
                    rate_limit_attempts,
                    max_retries,
                )
                if rate_limit_attempts >= max_retries:
                    log.error("APIStatusError %d: max retries exceeded", e.status_code)
                    raise
                if on_retry:
                    on_retry()
                time.sleep(5)
            else:
                log.error("APIStatusError %d: %s", e.status_code, e)
                raise
        except httpx.TimeoutException as e:
            rate_limit_attempts += 1
            log.warning("httpx.%s (attempt %d/%d) — retrying in 5s", type(e).__name__, rate_limit_attempts, max_retries)
            if rate_limit_attempts >= max_retries:
                log.error("httpx timeout: max retries exceeded")
                raise
            if on_retry:
                on_retry()
            time.sleep(5)


def _call_tool(tool_name: str, tool_input: dict) -> str:
    fn = TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        log.error("Unknown tool requested: %s", tool_name)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    log.info("Tool call → %s | args: %s", tool_name, list(tool_input.keys()))
    try:
        result = fn(**tool_input)
        preview = result[:200] + "…" if len(result) > 200 else result
        log.debug("Tool result ← %s | %s", tool_name, preview)
        return result
    except Exception as exc:
        log.error("Tool %s raised: %s", tool_name, exc, exc_info=True)
        return json.dumps({"error": str(exc)})


def _finalize(text_parts: list[str], collected: list[dict]) -> tuple[str, list[dict]]:
    text = "\n\n".join(t for t in text_parts if t)
    display_blocks = []
    if text:
        display_blocks.append({"type": "text", "text": text})
    for item in collected:
        if item["name"] == "group_and_count" and "data" in item["result"]:
            display_blocks.append(
                {
                    "type": "dataframe",
                    "data": pd.DataFrame(item["result"]["data"]),
                }
            )
    for item in collected:
        if "chart_path" in item["result"]:
            display_blocks.append({"type": "chart", "path": item["result"]["chart_path"]})
    return text, display_blocks


def _loop(
    client,
    model: str,
    tools: list,
    messages: list,
    tool_call_count: int,
    collected: list[dict],
    tool_names_called: list[str],
    on_retry=None,
) -> AgentResponse:
    """
    Inner loop shared by run_query() and continue_after_clarification().
    Mutates messages, collected, and tool_names_called in place.
    """
    log.debug("_loop start | tool_call_count=%d, history_len=%d", tool_call_count, len(messages))
    text_parts: list[str] = []

    while tool_call_count < MAX_TOOL_CALLS:
        log.debug("Sending to model (tool_calls_so_far=%d)", tool_call_count)
        response = _generate_with_retry(client, model, messages, tools, on_retry=on_retry)

        msg = response.choices[0].message

        # Accumulate any text the model emits (usually only on the final turn)
        if msg.content:
            text_parts.append(msg.content)

        # Serialise assistant message back into the history
        assistant_entry: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        log.debug(
            "Model response: content=%s, tool_calls=%s",
            bool(msg.content),
            [tc.function.name for tc in msg.tool_calls] if msg.tool_calls else [],
        )

        if not msg.tool_calls:
            text, display_blocks = _finalize(text_parts, collected)
            log.info(
                "Loop complete | tools_called=%s | display_blocks=%d | text_len=%d",
                tool_names_called,
                len(display_blocks),
                len(text),
            )
            # Strip system message from history before handing back to session
            conversation = [m for m in messages if m.get("role") != "system"]
            return AgentResponse(
                text=text,
                tool_calls=tool_names_called,
                display_blocks=display_blocks,
                clarification_question=None,
                history=conversation,
            )

        # ── Process tool calls ────────────────────────────────────────────────
        tool_responses: list[dict] = []
        clarify_call = None
        clarify_question_text = ""

        for tc in msg.tool_calls:
            tool_call_count += 1
            tool_names_called.append(tc.function.name)
            args = json.loads(tc.function.arguments)
            result_str = _call_tool(tc.function.name, args)
            result_data = json.loads(result_str)

            if tc.function.name == "clarify_question" and result_data.get("clarification_needed"):
                # Capture and handle after processing remaining calls in the batch
                clarify_call = tc
                clarify_question_text = result_data["question"]
            else:
                tool_responses.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    }
                )
                collected.append({"name": tc.function.name, "result": result_data})

        if clarify_call:
            # Add responses for any non-clarify tools in the same batch first
            messages.extend(tool_responses)
            log.info("Loop paused for clarification: %s", clarify_question_text)
            conversation = [m for m in messages if m.get("role") != "system"]
            return AgentResponse(
                text="",
                tool_calls=tool_names_called,
                display_blocks=[],
                clarification_question=clarify_question_text,
                history=conversation,
                _paused_messages=list(messages),
                _paused_tool_call_id=clarify_call.id,
                _paused_collected=list(collected),
                _paused_tool_call_count=tool_call_count,
            )

        messages.extend(tool_responses)

    # Hit tool cap — force a final answer
    log.warning("Tool call cap (%d) reached — forcing final answer", MAX_TOOL_CALLS)
    messages.append({"role": "user", "content": "Please provide a final answer now."})
    response = _generate_with_retry(client, model, messages, tools, on_retry=on_retry)
    msg = response.choices[0].message
    if msg.content:
        text_parts.append(msg.content)
    text, display_blocks = _finalize(text_parts, collected)
    conversation = [m for m in messages if m.get("role") != "system"]
    return AgentResponse(
        text=text,
        tool_calls=tool_names_called,
        display_blocks=display_blocks,
        clarification_question=None,
        history=conversation,
    )


def run_query(
    user_text: str,
    history: list | None = None,
    provider: str = "Groq",
    on_retry=None,
) -> AgentResponse:
    """
    Start a new agentic loop for a user query.

    Parameters
    ----------
    user_text : str
        The user's question.
    history : list | None
        Prior conversation turn dicts (no system message) from previous turns.
        Persist AgentResponse.history back to session state after each call.
    provider : str
        "Groq" or "Cerebras".
    """
    log.info("run_query | provider=%s | history_len=%d | query=%r", provider, len(history or []), user_text[:120])
    client, model = _build_client(provider)
    tools = _build_tools()

    messages: list = [_build_system_message()]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_text})

    return _loop(client, model, tools, messages, 0, [], [], on_retry=on_retry)


def continue_after_clarification(
    user_answer: str,
    paused: AgentResponse,
    provider: str = "Groq",
    on_retry=None,
) -> AgentResponse:
    """
    Resume the agentic loop after the user has answered a clarification question.

    The user's answer is injected as a tool response for the paused
    clarify_question call — the model receives full context and continues
    from exactly where it left off.

    Parameters
    ----------
    user_answer : str
        What the user typed in response to the clarification question.
    paused : AgentResponse
        The AgentResponse that had clarification_question set.
    provider : str
        "Groq" or "Cerebras". Can differ from the provider that asked the question.
    """
    log.info("continue_after_clarification | provider=%s | answer=%r", provider, user_answer[:120])
    client, model = _build_client(provider)
    tools = _build_tools()

    messages = list(paused._paused_messages)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": paused._paused_tool_call_id,
            "content": json.dumps({"user_answer": user_answer}),
        }
    )

    return _loop(
        client,
        model,
        tools,
        messages,
        paused._paused_tool_call_count,
        list(paused._paused_collected),
        list(paused.tool_calls),
        on_retry=on_retry,
    )
