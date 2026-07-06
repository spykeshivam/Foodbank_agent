"""
End-to-end agent query tests.

Runs 5 diverse natural-language queries through the full agentic loop
against both Groq and Cerebras, asserting every response is correctly structured.

Output format contract for every response:
  - AgentResponse returned (not an exception)
  - .error        : None (no failure)
  - .text         : non-empty string containing at least one digit
  - .display_blocks : list with at least one block
  - each block has a valid "type" key in {"text", "dataframe", "chart"}
  - "text"      blocks: .text is a non-empty string
  - "dataframe" blocks: .data is a non-empty pd.DataFrame
  - "chart"     blocks: .path points to a readable Plotly JSON file

Run with:
    uv run pytest tests/test_agent_queries.py -v -s
    uv run pytest tests/test_agent_queries.py -v -s -k Groq
    uv run pytest tests/test_agent_queries.py -v -s -k Cerebras
"""

import json
import os
import sys
import time

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import PROVIDERS, AgentResponse, run_query
from config import CREDENTIALS_FILE
from sheets import fetch_logins, fetch_registrations

# ── Skip entire module if credentials missing ─────────────────────────────────
pytestmark = pytest.mark.skipif(
    not os.path.exists(CREDENTIALS_FILE),
    reason="credentials.json not found — skipping agent query tests",
)


def _provider_available(provider: str) -> bool:
    key_env = PROVIDERS[provider]["api_key_env"]
    return bool(os.environ.get(key_env))


# ── Load real data once for the whole test session ────────────────────────────
@pytest.fixture(scope="session")
def real_data():
    reg = pd.DataFrame(fetch_registrations())
    logins = pd.DataFrame(fetch_logins())
    return reg, logins


@pytest.fixture(autouse=True)
def rate_limit_pause(request):
    """Pause between tests to respect free-tier RPM limits."""
    if getattr(request.session, "_query_test_count", 0) > 0:
        time.sleep(30)
    request.session._query_test_count = getattr(request.session, "_query_test_count", 0) + 1


# ── Shared format checker ─────────────────────────────────────────────────────
def assert_valid_response(result: AgentResponse, query: str) -> None:
    assert isinstance(result, AgentResponse), f"Expected AgentResponse, got {type(result)} for: {query!r}"
    assert result.error is None, f"Agent returned an error: {result.error!r}\nQuery: {query!r}"
    assert result.text.strip(), f"Response text is empty for: {query!r}"
    assert any(ch.isdigit() for ch in result.text), (
        f"Response contains no numerical result for: {query!r}\nText: {result.text}"
    )
    assert isinstance(result.display_blocks, list), f"display_blocks is not a list for: {query!r}"
    assert len(result.display_blocks) >= 1, f"display_blocks is empty for: {query!r}"

    valid_types = {"text", "dataframe", "chart"}
    for i, block in enumerate(result.display_blocks):
        assert "type" in block, f"Block {i} missing 'type' key for: {query!r}"
        assert block["type"] in valid_types, f"Block {i} has unknown type {block['type']!r} for: {query!r}"

        if block["type"] == "text":
            assert isinstance(block.get("text"), str) and block["text"].strip(), (
                f"Text block {i} has empty/missing text for: {query!r}"
            )
        elif block["type"] == "dataframe":
            df = block.get("data")
            assert isinstance(df, pd.DataFrame), f"Dataframe block {i} 'data' is not a DataFrame for: {query!r}"
            assert not df.empty, f"Dataframe block {i} is empty for: {query!r}"
        elif block["type"] == "chart":
            path = block.get("path")
            assert isinstance(path, str), f"Chart block {i} 'path' is not a string for: {query!r}"
            assert os.path.exists(path), f"Chart file does not exist at {path!r} for: {query!r}"
            with open(path) as f:
                chart_json = json.load(f)
            assert "data" in chart_json, f"Chart JSON at {path!r} missing 'data' key for: {query!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# Parametrized test class — runs every query against Groq AND Cerebras
# ═══════════════════════════════════════════════════════════════════════════════

PROVIDERS_TO_TEST = ["Groq", "Cerebras"]


@pytest.mark.parametrize("provider", PROVIDERS_TO_TEST)
class TestAgentQueries:
    def _skip_if_unavailable(self, provider: str):
        if not _provider_available(provider):
            pytest.skip(f"{PROVIDERS[provider]['api_key_env']} not set — skipping {provider} tests")

    def test_q1_simple_count(self, provider, real_data):
        """Simple count — filter and numerical answer."""
        self._skip_if_unavailable(provider)
        registrations, logins = real_data
        query = "How many people have Halal dietary requirements?"
        result = run_query(query, registrations, logins, provider=provider)
        assert_valid_response(result, query)
        assert "halal" in result.text.lower(), f"Expected 'halal' in response, got: {result.text}"

    def test_q2_pie_chart(self, provider, real_data):
        """Breakdown query expecting a pie chart."""
        self._skip_if_unavailable(provider)
        registrations, logins = real_data
        query = "Show me the gender breakdown of registered users as a pie chart."
        result = run_query(query, registrations, logins, provider=provider)
        assert_valid_response(result, query)
        chart_blocks = [b for b in result.display_blocks if b["type"] == "chart"]
        assert len(chart_blocks) >= 1, "Expected at least one chart block"
        text_lower = result.text.lower()
        assert "male" in text_lower or "female" in text_lower, f"Gender not mentioned in: {result.text}"

    def test_q3_time_series_bar_chart(self, provider, real_data):
        """Monthly login trend with a bar chart."""
        self._skip_if_unavailable(provider)
        registrations, logins = real_data
        query = "How many logins were there each month? Show a bar chart."
        result = run_query(query, registrations, logins, provider=provider)
        assert_valid_response(result, query)
        chart_blocks = [b for b in result.display_blocks if b["type"] == "chart"]
        assert len(chart_blocks) >= 1, "Expected a bar chart for monthly logins"

    def test_q4_join_and_filter(self, provider, real_data):
        """Cross-sheet join with multiple filters."""
        self._skip_if_unavailable(provider)
        registrations, logins = real_data
        query = "How many male users visited on a Tuesday?"
        result = run_query(query, registrations, logins, provider=provider)
        assert_valid_response(result, query)

    def test_q5_language_breakdown_with_table(self, provider, real_data):
        """Top-N categorical breakdown expecting a table and chart."""
        self._skip_if_unavailable(provider)
        registrations, logins = real_data
        query = "What are the top spoken languages among registered users? Show a chart."
        result = run_query(query, registrations, logins, provider=provider)
        assert_valid_response(result, query)
        rich_blocks = [b for b in result.display_blocks if b["type"] in ("dataframe", "chart")]
        assert len(rich_blocks) >= 1, "Expected at least a table or chart for language breakdown"
        known_languages = ["english", "bengali", "arabic", "lithuanian", "somali"]
        text_lower = result.text.lower()
        assert any(lang in text_lower for lang in known_languages), f"No known language mentioned in: {result.text}"
