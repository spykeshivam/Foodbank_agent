"""
Core agentic loop — no Streamlit dependency.

app.py and tests both import from here.
"""
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import httpx
import pandas as pd
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from log_config import get_logger, setup_logging
from tool_schemas import TOOL_SCHEMAS
from tools import TOOL_FUNCTIONS

setup_logging()
log = get_logger(__name__)

MAX_TOOL_CALLS = 10

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


@dataclass
class AgentResponse:
    """Structured result returned by run_query() and continue_after_clarification()."""
    text: str                               # final natural-language answer
    tool_calls: list[str]                   # ordered list of tool names called
    display_blocks: list[dict]              # [{type, ...}] ready for rendering
    clarification_question: str | None      # non-None if agent paused to clarify
    history: list = field(default_factory=list)  # full message history — persist in session

    # ── Internal resume state (populated only when clarification_question is set) ──
    # app.py passes the whole AgentResponse back to continue_after_clarification();
    # these fields carry exactly where the loop was when it paused.
    _paused_messages: list = field(default_factory=list)
    _paused_fc_name: str = ""               # name of the clarify_question call
    _paused_collected: list = field(default_factory=list)
    _paused_tool_call_count: int = 0


def _build_tool() -> types.Tool:
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=s["name"],
                description=s["description"],
                parameters_json_schema=s["parameters"],
            )
            for s in TOOL_SCHEMAS
        ]
    )


def _build_config() -> types.GenerateContentConfig:
    today = datetime.now().strftime("%A, %d %B %Y")
    system_prompt = f"Today's date is {today}.\n\n{SYSTEM_PROMPT}"
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[_build_tool()],
        temperature=0.2,
        http_options=types.HttpOptions(timeout=600_000),  # 600s in milliseconds
    )


def _generate_with_retry(client, model, contents, config, max_retries=5, on_retry=None):
    """Call generate_content with exponential backoff on 429 and fixed retry on 503/504."""
    delay = 60
    rate_limit_attempts = 0
    attempt_total = 0
    while True:
        attempt_total += 1
        log.debug("API call attempt %d (model=%s, messages=%d)", attempt_total, model, len(contents))
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config,
            )
            log.debug("API call succeeded on attempt %d", attempt_total)
            return response
        except genai_errors.ClientError as e:
            if e.code == 429:
                rate_limit_attempts += 1
                log.warning("429 rate-limit hit (attempt %d/%d) — waiting %ds", rate_limit_attempts, max_retries, delay)
                if rate_limit_attempts >= max_retries:
                    log.error("429 rate-limit: max retries exceeded")
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 300)
            else:
                log.error("ClientError %s: %s", e.code, e)
                raise
        except genai_errors.ServerError as e:
            if e.status in ("UNAVAILABLE", "DEADLINE_EXCEEDED"):
                rate_limit_attempts += 1
                log.warning("ServerError %s (attempt %d/%d) — retrying in 5s", e.status, rate_limit_attempts, max_retries)
                if rate_limit_attempts >= max_retries:
                    log.error("ServerError %s: max retries exceeded", e.status)
                    raise
                if on_retry:
                    on_retry()
                time.sleep(5)
            else:
                log.error("ServerError %s: %s", e.status, e)
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


def _args_to_dict(args) -> dict:
    if args is None:
        return {}
    result = {}
    for k, v in args.items():
        if hasattr(v, "items"):
            result[k] = dict(v)
        elif hasattr(v, "__iter__") and not isinstance(v, str):
            result[k] = list(v)
        else:
            result[k] = v
    return result


def _call_tool(tool_name: str, tool_input: dict) -> str:
    fn = TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        log.error("Unknown tool requested: %s", tool_name)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    log.info("Tool call → %s | args: %s", tool_name, list(tool_input.keys()))
    try:
        result = fn(**tool_input)
        # Log a short preview of the result (avoid flooding logs with large dataframes)
        preview = result[:200] + "…" if len(result) > 200 else result
        log.debug("Tool result ← %s | %s", tool_name, preview)
        return result
    except Exception as exc:
        log.error("Tool %s raised: %s", tool_name, exc, exc_info=True)
        return json.dumps({"error": str(exc)})


def _finalize(text_blocks: list[str], collected: list[dict]) -> tuple[str, list[dict]]:
    text = "\n\n".join(text_blocks)
    display_blocks = []
    if text:
        display_blocks.append({"type": "text", "text": text})
    for item in collected:
        if item["name"] == "group_and_count" and "data" in item["result"]:
            display_blocks.append({
                "type": "dataframe",
                "data": pd.DataFrame(item["result"]["data"]),
            })
    for item in collected:
        if "chart_path" in item["result"]:
            display_blocks.append({"type": "chart", "path": item["result"]["chart_path"]})
    return text, display_blocks


def _loop(
    client,
    config,
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

    while tool_call_count < MAX_TOOL_CALLS:
        log.debug("Sending to model (tool_calls_so_far=%d)", tool_call_count)
        response = _generate_with_retry(client, "gemini-2.5-flash", messages, config, on_retry=on_retry)

        candidate = response.candidates[0]
        messages.append(candidate.content)

        text_parts = [p.text for p in candidate.content.parts if p.text]
        function_calls = [
            p.function_call
            for p in candidate.content.parts
            if p.function_call and p.function_call.name
        ]

        log.debug(
            "Model response: %d text part(s), %d function call(s): %s",
            len(text_parts),
            len(function_calls),
            [fc.name for fc in function_calls],
        )

        if not function_calls:
            text, display_blocks = _finalize(text_parts, collected)
            log.info(
                "Loop complete | tools_called=%s | display_blocks=%d | text_len=%d",
                tool_names_called,
                len(display_blocks),
                len(text),
            )
            return AgentResponse(
                text=text,
                tool_calls=tool_names_called,
                display_blocks=display_blocks,
                clarification_question=None,
                history=list(messages),
            )

        tool_responses: list[types.Part] = []

        for fc in function_calls:
            tool_call_count += 1
            tool_names_called.append(fc.name)
            args = _args_to_dict(fc.args)
            result_str = _call_tool(fc.name, args)
            result_data = json.loads(result_str)

            if fc.name == "clarify_question" and result_data.get("clarification_needed"):
                log.info("Loop paused for clarification: %s", result_data["question"])
                return AgentResponse(
                    text="",
                    tool_calls=tool_names_called,
                    display_blocks=[],
                    clarification_question=result_data["question"],
                    history=list(messages),
                    _paused_messages=list(messages),
                    _paused_fc_name=fc.name,
                    _paused_collected=list(collected),
                    _paused_tool_call_count=tool_call_count,
                )

            tool_responses.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result_str},
                )
            )
            collected.append({"name": fc.name, "result": result_data})

        messages.append(types.Content(role="user", parts=tool_responses))

    # Hit tool cap — force a final answer
    log.warning("Tool call cap (%d) reached — forcing final answer", MAX_TOOL_CALLS)
    messages.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text="Please provide a final answer now.")],
    ))
    response = _generate_with_retry(client, "gemini-2.5-flash", messages, config, on_retry=on_retry)
    text_parts = [p.text for p in response.candidates[0].content.parts if p.text]
    text, display_blocks = _finalize(text_parts, collected)
    return AgentResponse(
        text=text,
        tool_calls=tool_names_called,
        display_blocks=display_blocks,
        clarification_question=None,
        history=list(messages),
    )


def run_query(
    user_text: str,
    history: list | None = None,
    on_retry=None,
) -> AgentResponse:
    """
    Start a new agentic loop for a user query.

    Parameters
    ----------
    user_text : str
        The user's question.
    history : list | None
        Prior Gemini Content objects from previous turns in the session.
        Persist AgentResponse.history back to session state after each call.
    """
    log.info("run_query | history_len=%d | query=%r", len(history or []), user_text[:120])
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = _build_config()

    messages: list = list(history or [])
    messages.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))

    return _loop(client, config, messages, 0, [], [], on_retry=on_retry)


def continue_after_clarification(
    user_answer: str,
    paused: AgentResponse,
    on_retry=None,
) -> AgentResponse:
    """
    Resume the agentic loop after the user has answered a clarification question.

    The user's answer is injected as a tool_result for the paused
    clarify_question call — the model receives full context and continues
    from exactly where it left off.

    Parameters
    ----------
    user_answer : str
        What the user typed in response to the clarification question.
    paused : AgentResponse
        The AgentResponse that had clarification_question set.
    """
    log.info("continue_after_clarification | answer=%r", user_answer[:120])
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = _build_config()

    messages = list(paused._paused_messages)
    messages.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=paused._paused_fc_name,
                    response={"user_answer": user_answer},
                )
            ],
        )
    )

    return _loop(
        client,
        config,
        messages,
        paused._paused_tool_call_count,
        list(paused._paused_collected),
        list(paused.tool_calls),
        on_retry=on_retry,
    )
