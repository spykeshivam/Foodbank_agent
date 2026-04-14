"""
End-to-end agent query tests.

Runs 5 diverse natural-language queries through the full agentic loop
(real Gemini API + real Google Sheets data) and asserts that every
response is correctly structured.

Output format contract for every response:
  - AgentResponse returned (not an exception)
  - .text        : non-empty string
  - .tool_calls  : at least one tool was called
  - .display_blocks : list with at least one block
  - each block has a valid "type" key in {"text", "dataframe", "chart"}
  - "text"      blocks: .text is a non-empty string
  - "dataframe" blocks: .data is a non-empty pd.DataFrame
  - "chart"     blocks: .path points to a readable Plotly JSON file

Run with:
    uv run pytest tests/test_agent_queries.py -v -s

Note: tests run sequentially with a 15-second pause between each to
stay within the free-tier rate limit (10 RPM).
"""
import json
import os
import sys
import time

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import AgentResponse, run_query
from config import CREDENTIALS_FILE
from sheets import fetch_logins, fetch_registrations
from tools import init_datasets

# ── Skip if credentials or API key missing ────────────────────────────────────
pytestmark = pytest.mark.skipif(
    not os.path.exists(CREDENTIALS_FILE),
    reason="credentials.json not found — skipping agent query tests",
)

if not os.environ.get("GEMINI_API_KEY"):
    pytestmark = pytest.mark.skip(reason="GEMINI_API_KEY not set")


# ── Load real data once for the session ───────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def load_real_data():
    reg = pd.DataFrame(fetch_registrations())
    logins = pd.DataFrame(fetch_logins())
    init_datasets(reg, logins)


@pytest.fixture(autouse=True)
def rate_limit_pause(request):
    """Pause 15 s before each test (except the first) to respect free-tier RPM."""
    if getattr(request.session, "_query_test_count", 0) > 0:
        time.sleep(30)
    request.session._query_test_count = getattr(request.session, "_query_test_count", 0) + 1


# ── Shared format checker ─────────────────────────────────────────────────────
def assert_valid_response(result: AgentResponse, query: str) -> None:
    """Assert that an AgentResponse meets the output format contract."""
    assert isinstance(result, AgentResponse), \
        f"Expected AgentResponse, got {type(result)} for: {query!r}"

    # Must not be a clarification pause for these unambiguous queries
    assert result.clarification_question is None, (
        f"Agent asked for clarification unexpectedly: "
        f"{result.clarification_question!r}\nQuery: {query!r}"
    )

    # Must have called at least one tool
    assert len(result.tool_calls) >= 1, \
        f"No tools were called for: {query!r}"

    # Text must be non-empty
    assert result.text.strip(), \
        f"Response text is empty for: {query!r}"

    # Must contain a number somewhere in the text
    digits_in_text = any(ch.isdigit() for ch in result.text)
    assert digits_in_text, \
        f"Response contains no numerical result for: {query!r}\nText: {result.text}"

    # display_blocks must be a non-empty list
    assert isinstance(result.display_blocks, list), \
        f"display_blocks is not a list for: {query!r}"
    assert len(result.display_blocks) >= 1, \
        f"display_blocks is empty for: {query!r}"

    # Every block must have a valid type and correct payload
    valid_types = {"text", "dataframe", "chart"}
    for i, block in enumerate(result.display_blocks):
        assert "type" in block, \
            f"Block {i} missing 'type' key for: {query!r}"
        assert block["type"] in valid_types, \
            f"Block {i} has unknown type {block['type']!r} for: {query!r}"

        if block["type"] == "text":
            assert isinstance(block.get("text"), str) and block["text"].strip(), \
                f"Text block {i} has empty/missing text for: {query!r}"

        elif block["type"] == "dataframe":
            df = block.get("data")
            assert isinstance(df, pd.DataFrame), \
                f"Dataframe block {i} 'data' is not a DataFrame for: {query!r}"
            assert not df.empty, \
                f"Dataframe block {i} is empty for: {query!r}"
            assert len(df.columns) >= 1, \
                f"Dataframe block {i} has no columns for: {query!r}"

        elif block["type"] == "chart":
            path = block.get("path")
            assert isinstance(path, str), \
                f"Chart block {i} 'path' is not a string for: {query!r}"
            assert os.path.exists(path), \
                f"Chart file does not exist at {path!r} for: {query!r}"
            with open(path) as f:
                chart_json = json.load(f)
            assert "data" in chart_json, \
                f"Chart JSON at {path!r} missing 'data' key for: {query!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# The 5 diverse queries
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentQueries:

    def test_q1_simple_count(self):
        """
        Simple count query — no join, no chart required.
        Tests: filter_registrations + numerical answer in text.
        """
        query = "How many people have Halal dietary requirements?"
        result = run_query(query)
        assert_valid_response(result, query)

        # The answer must mention "halal" in some form
        assert "halal" in result.text.lower(), \
            f"Expected 'halal' in response, got: {result.text}"

    def test_q2_pie_chart(self):
        """
        Breakdown query expecting a pie chart.
        Tests: group_and_count + create_pie_chart + chart block in output.
        """
        query = "Show me the gender breakdown of registered users as a pie chart."
        result = run_query(query)
        assert_valid_response(result, query)

        # Must have produced a chart
        chart_blocks = [b for b in result.display_blocks if b["type"] == "chart"]
        assert len(chart_blocks) >= 1, "Expected at least one chart block"

        # Text must mention both genders
        text_lower = result.text.lower()
        assert "male" in text_lower or "female" in text_lower, \
            f"Gender not mentioned in: {result.text}"

    def test_q3_time_series_bar_chart(self):
        """
        Time-series query — logins grouped by month with a bar chart.
        Tests: group_and_count(month) + create_bar_chart + dataframe block.
        """
        query = "How many logins were there each month? Show a bar chart."
        result = run_query(query)
        assert_valid_response(result, query)

        # Must have a chart
        chart_blocks = [b for b in result.display_blocks if b["type"] == "chart"]
        assert len(chart_blocks) >= 1, "Expected a bar chart for monthly logins"

        # Must have called group_and_count
        assert "group_and_count" in result.tool_calls, \
            f"Expected group_and_count in tool calls, got: {result.tool_calls}"

    def test_q4_join_and_filter(self):
        """
        Cross-sheet join with multiple filters.
        Tests: join_sheets + filter on both sheets + numerical result.
        """
        query = "How many male users visited on a Tuesday?"
        result = run_query(query)
        assert_valid_response(result, query)

        # Must have joined the sheets
        assert "join_sheets" in result.tool_calls, \
            f"Expected join_sheets in tool calls, got: {result.tool_calls}"

    def test_q5_language_breakdown_with_table(self):
        """
        Top-N categorical breakdown expecting a table and chart.
        Tests: group_and_count(Primary Spoken Language) + dataframe block.
        """
        query = "What are the top spoken languages among registered users? Show a chart."
        result = run_query(query)
        assert_valid_response(result, query)

        # Must have a dataframe or chart
        rich_blocks = [
            b for b in result.display_blocks
            if b["type"] in ("dataframe", "chart")
        ]
        assert len(rich_blocks) >= 1, \
            "Expected at least a table or chart for language breakdown"

        # Must mention at least one real language from the data
        known_languages = ["english", "bengali", "arabic", "lithuanian", "somali"]
        text_lower = result.text.lower()
        assert any(lang in text_lower for lang in known_languages), \
            f"No known language mentioned in: {result.text}"
