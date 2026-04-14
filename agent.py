"""
Core agentic loop — no Streamlit dependency.

app.py and tests both import from here.
"""
import json
import os
import time
from dataclasses import dataclass

import pandas as pd
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from tool_schemas import TOOL_SCHEMAS
from tools import TOOL_FUNCTIONS

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
    """Structured result returned by run_query()."""
    text: str                               # final natural-language answer
    tool_calls: list[str]                   # ordered list of tool names called
    display_blocks: list[dict]              # [{type, ...}] ready for rendering
    clarification_question: str | None      # non-None if agent paused to ask


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
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[_build_tool()],
        temperature=0.2,
    )


def _generate_with_retry(client, model, contents, config, max_retries=5):
    """Call generate_content with exponential backoff on 429 rate-limit errors."""
    delay = 60  # start at 60s — generous enough for RPM limits
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except genai_errors.ClientError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 300)  # cap at 5 minutes
            else:
                raise
    raise RuntimeError("Max retries exceeded")


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
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return fn(**tool_input)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _finalize(text_blocks: list[str], collected: list[dict]) -> tuple[str, list[dict]]:
    """Build final text and display_blocks from accumulated results."""
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


def run_query(
    user_text: str,
    history: list | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> AgentResponse:
    """
    Run the full agentic loop for a single user query.

    Parameters
    ----------
    user_text : str
        The user's question.
    history : list | None
        Prior Gemini Content objects (for multi-turn conversations).
        Pass None to start a fresh conversation.
    max_tool_calls : int
        Hard cap on tool calls before forcing a final answer.

    Returns
    -------
    AgentResponse
        .text                   — final natural-language answer
        .tool_calls             — names of every tool called
        .display_blocks         — list of {type, ...} render blocks
        .clarification_question — set when agent needs more info from user
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = _build_config()

    messages: list = list(history or [])
    messages.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))

    tool_call_count = 0
    collected: list[dict] = []
    tool_names_called: list[str] = []

    while tool_call_count < max_tool_calls:
        response = _generate_with_retry(client, "gemini-2.5-flash", messages, config)

        candidate = response.candidates[0]
        messages.append(candidate.content)

        text_parts = [p.text for p in candidate.content.parts if p.text]
        function_calls = [
            p.function_call
            for p in candidate.content.parts
            if p.function_call and p.function_call.name
        ]

        if not function_calls:
            text, display_blocks = _finalize(text_parts, collected)
            return AgentResponse(
                text=text,
                tool_calls=tool_names_called,
                display_blocks=display_blocks,
                clarification_question=None,
            )

        tool_responses: list[types.Part] = []

        for fc in function_calls:
            tool_call_count += 1
            tool_names_called.append(fc.name)
            args = _args_to_dict(fc.args)
            result_str = _call_tool(fc.name, args)
            result_data = json.loads(result_str)

            if fc.name == "clarify_question" and result_data.get("clarification_needed"):
                return AgentResponse(
                    text="",
                    tool_calls=tool_names_called,
                    display_blocks=[],
                    clarification_question=result_data["question"],
                )

            tool_responses.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result_str},
                )
            )
            collected.append({"name": fc.name, "result": result_data})

        messages.append(types.Content(role="user", parts=tool_responses))

    # Forced final answer after hitting tool cap
    messages.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text="Please provide a final answer now.")],
    ))
    response = _generate_with_retry(client, "gemini-2.5-flash", messages, config)
    text_parts = [p.text for p in response.candidates[0].content.parts if p.text]
    text, display_blocks = _finalize(text_parts, collected)
    return AgentResponse(
        text=text,
        tool_calls=tool_names_called,
        display_blocks=display_blocks,
        clarification_question=None,
    )
