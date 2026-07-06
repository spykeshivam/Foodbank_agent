"""
Code-generation agent — replaces the 10-tool agentic loop.

Flow (mirrors the PandasAI pattern):
  user question
    → build prompt with DataFrame schema (no raw data sent to LLM)
    → LLM generates a pandas/plotly code snippet
    → execute locally against the real DataFrames
    → return result (number / table / chart) to Streamlit
"""

import os
import re
import tempfile
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import httpx
import openai
import pandas as pd
import plotly.express as px

from log_config import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)

# ── Provider config ───────────────────────────────────────────────────────────
PROVIDERS: dict[str, dict] = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "Cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "model": "llama3.3-70b",
    },
}

MAX_CODE_RETRIES = 2  # re-ask LLM to fix broken code this many times

_api_call_times: deque[float] = deque()
_RATE_WINDOW = 300


# ── Response dataclass ────────────────────────────────────────────────────────
@dataclass
class AgentResponse:
    text: str  # natural-language answer (may be empty if chart/table speaks for itself)
    display_blocks: list[dict]  # [{type: text|dataframe|chart, ...}] for Streamlit
    history: list = field(default_factory=list)  # conversation turns — persist in session
    error: str | None = None  # set if all retries failed


# ── LLM client helpers ────────────────────────────────────────────────────────
def _record_api_call() -> None:
    now = time.monotonic()
    _api_call_times.append(now)
    cutoff = now - _RATE_WINDOW
    while _api_call_times and _api_call_times[0] < cutoff:
        _api_call_times.popleft()
    log.info("API requests in last 5 min: %d", len(_api_call_times))


def _build_client(provider: str) -> tuple[openai.OpenAI, str]:
    cfg = PROVIDERS[provider]
    return openai.OpenAI(
        base_url=cfg["base_url"],
        api_key=os.environ[cfg["api_key_env"]],
    ), cfg["model"]


def _call_llm(
    client: openai.OpenAI,
    model: str,
    messages: list,
    on_retry=None,
    max_retries: int = 5,
) -> str:
    """Call the LLM and return the response text. Retries on rate limits and server errors."""
    attempts = 0
    while True:
        attempts += 1
        try:
            _record_api_call()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                timeout=120,
            )
            u = response.usage
            log.info("Tokens — prompt: %d, output: %d", u.prompt_tokens, u.completion_tokens)
            return response.choices[0].message.content
        except openai.RateLimitError:
            log.warning("429 rate-limit (attempt %d/%d) — waiting 40s", attempts, max_retries)
            if attempts >= max_retries:
                raise
            time.sleep(40)
        except openai.APIStatusError as e:
            if e.status_code in (503, 504):
                log.warning("APIStatusError %d (attempt %d/%d) — retrying in 5s", e.status_code, attempts, max_retries)
                if attempts >= max_retries:
                    raise
                if on_retry:
                    on_retry()
                time.sleep(5)
            else:
                raise
        except httpx.TimeoutException:
            log.warning("Timeout (attempt %d/%d) — retrying in 5s", attempts, max_retries)
            if attempts >= max_retries:
                raise
            if on_retry:
                on_retry()
            time.sleep(5)


# ── Prompt construction ───────────────────────────────────────────────────────
def _schema_block(df: pd.DataFrame, name: str) -> str:
    """Compact schema: shape + columns with dtypes + 3 sample rows."""
    col_info = "\n".join(f"  - {col}: {df[col].dtype}" for col in df.columns)
    sample = df.head(3).to_string(index=False)
    return f"{name} ({len(df):,} rows)\nColumns:\n{col_info}\nSample:\n{sample}"


_SYSTEM_TEMPLATE = """\
You are a data analyst for St Dunstan's Food Bank. Today is {today}.

You have two pandas DataFrames already loaded in memory:

{reg_schema}

{login_schema}

The two DataFrames join on the "Username" column.

YOUR JOB: Write a single Python code block to answer the user's question.

RULES:
- Available variables: `registrations`, `logins`, `pd` (pandas), `px` (plotly.express)
- Do NOT import anything — everything is already available
- Do NOT use print()
- For a number or text answer: assign a descriptive sentence to `result`
  e.g.  result = f"{{count}} people have Halal dietary requirements"
- For a chart: assign a Plotly figure to `fig`
- For a table: assign a DataFrame to `result`
- You may set both `result` and `fig` when useful
- Return ONLY a ```python code block — no explanation outside it

EXAMPLES:

Q: How many women are registered?
```python
count = len(registrations[registrations["Sex"].str.strip().str.lower() == "female"])
result = f"{{count}} women are registered."
```

Q: Show logins per month as a bar chart.
```python
logins["Month"] = pd.to_datetime(logins["Timestamp"]).dt.to_period("M").astype(str)
monthly = logins.groupby("Month").size().reset_index(name="Logins")
fig = px.bar(monthly, x="Month", y="Logins", title="Logins per Month")
result = monthly
```

Q: What are the top 5 spoken languages?
```python
top = registrations["Primary Spoken Language"].value_counts().head(5).reset_index()
top.columns = ["Language", "Count"]
fig = px.bar(top, x="Language", y="Count", title="Top 5 Spoken Languages")
result = top
```\
"""


def _build_messages(
    question: str,
    registrations: pd.DataFrame,
    logins: pd.DataFrame,
    history: list,
    error_feedback: str | None = None,
) -> list[dict]:
    today = datetime.now().strftime("%A, %d %B %Y")
    system_content = _SYSTEM_TEMPLATE.format(
        today=today,
        reg_schema=_schema_block(registrations, "registrations"),
        login_schema=_schema_block(logins, "logins"),
    )
    messages = [{"role": "system", "content": system_content}]
    messages.extend(history)

    if error_feedback:
        user_content = f"{question}\n\nYour previous code raised an error — please fix it:\n```\n{error_feedback}\n```"
    else:
        user_content = question

    messages.append({"role": "user", "content": user_content})
    return messages


# ── Code extraction & execution ───────────────────────────────────────────────
def _extract_code(text: str) -> str:
    """Pull the Python code block out of the LLM response."""
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _execute(
    code: str,
    registrations: pd.DataFrame,
    logins: pd.DataFrame,
) -> tuple[dict, str | None]:
    """
    Run code in a restricted namespace.
    Returns ({result, fig}, error_traceback_or_None).
    """
    namespace = {
        "pd": pd,
        "px": px,
        "registrations": registrations.copy(),
        "logins": logins.copy(),
    }
    try:
        exec(code, namespace)  # noqa: S102
        return {k: namespace[k] for k in ("result", "fig") if k in namespace and namespace[k] is not None}, None
    except Exception:
        return {}, traceback.format_exc()


# ── Result → display blocks ───────────────────────────────────────────────────
def _to_display_blocks(output: dict) -> tuple[str, list[dict]]:
    blocks: list[dict] = []
    text = ""

    result = output.get("result")
    fig = output.get("fig")

    if result is not None:
        if isinstance(result, pd.DataFrame) and not result.empty:
            blocks.append({"type": "dataframe", "data": result})
        else:
            text = str(result)
            blocks.append({"type": "text", "text": text})

    if fig is not None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = tmp.name
        fig.write_json(path)
        blocks.append({"type": "chart", "path": path})

    return text, blocks


# ── Public API ────────────────────────────────────────────────────────────────
def run_query(
    question: str,
    registrations: pd.DataFrame,
    logins: pd.DataFrame,
    history: list | None = None,
    provider: str = "Groq",
    on_retry=None,
) -> AgentResponse:
    """
    Answer a natural-language question about the two DataFrames.

    Parameters
    ----------
    question      : user's question in plain English
    registrations : registrations DataFrame
    logins        : logins DataFrame
    history       : prior conversation turns (no system message) — pass back
                    AgentResponse.history after each call for multi-turn context
    provider      : "Groq" or "Cerebras"
    on_retry      : optional callback shown to user when server is busy
    """
    log.info("run_query | provider=%s | question=%r", provider, question[:120])
    client, model = _build_client(provider)
    history = list(history or [])
    error_feedback: str | None = None

    for attempt in range(MAX_CODE_RETRIES + 1):
        messages = _build_messages(question, registrations, logins, history, error_feedback)
        response_text = _call_llm(client, model, messages, on_retry=on_retry)
        log.debug("LLM raw response (attempt %d):\n%s", attempt + 1, response_text)

        code = _extract_code(response_text)
        log.info("Executing code (attempt %d):\n%s", attempt + 1, code)

        output, error = _execute(code, registrations, logins)

        if error:
            log.warning("Code error (attempt %d/%d):\n%s", attempt + 1, MAX_CODE_RETRIES + 1, error)
            error_feedback = error
            continue

        if not output:
            log.warning("Code ran but produced no result or fig (attempt %d)", attempt + 1)
            error_feedback = (
                "The code ran without errors but did not assign anything to `result` or `fig`. Please fix it."
            )
            continue

        # ── Success ───────────────────────────────────────────────────────────
        text, display_blocks = _to_display_blocks(output)

        new_history = list(history)
        new_history.append({"role": "user", "content": question})
        new_history.append({"role": "assistant", "content": response_text})

        log.info("run_query success | blocks=%d | text=%r", len(display_blocks), text[:80])
        return AgentResponse(text=text, display_blocks=display_blocks, history=new_history)

    # ── All retries exhausted ─────────────────────────────────────────────────
    log.error("run_query failed after %d attempts", MAX_CODE_RETRIES + 1)
    return AgentResponse(
        text="",
        display_blocks=[],
        history=history,
        error=f"Could not produce a valid answer after {MAX_CODE_RETRIES + 1} attempts.\n\n{error_feedback}",
    )
