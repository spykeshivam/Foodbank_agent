"""
Unit tests for agent.py internal helpers — no credentials, no network.

Covers:
  - _extract_code   : parsing LLM response text into a code string
  - _execute        : running generated code in a sandboxed namespace
  - _to_display_blocks : converting exec output to display blocks
"""

import json
import os
import sys

import pandas as pd
import plotly.express as px
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import _execute, _extract_code, _to_display_blocks

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def small_reg() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Username": ["alice", "bob"],
            "Sex": ["Female", "Male"],
            "Primary Spoken Language": ["English", "Bengali"],
        }
    )


@pytest.fixture()
def small_logins() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Username": ["alice", "alice", "bob"],
            "Timestamp": ["2024-01-05 10:00", "2024-02-10 11:00", "2024-01-20 09:00"],
            "Day": ["Friday", "Saturday", "Saturday"],
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# _extract_code
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractCode:
    def test_python_tagged_block(self):
        text = "Here is my answer:\n```python\nresult = 42\n```\nDone."
        assert _extract_code(text) == "result = 42"

    def test_untagged_block_fallback(self):
        text = "Sure:\n```\nresult = 'hello'\n```"
        assert _extract_code(text) == "result = 'hello'"

    def test_prefers_python_tagged_over_untagged(self):
        text = "```\nresult = 'untagged'\n```\n```python\nresult = 'tagged'\n```"
        assert _extract_code(text) == "result = 'tagged'"

    def test_raw_text_fallback_when_no_backticks(self):
        raw = "result = len(registrations)"
        assert _extract_code(raw) == raw

    def test_empty_string_returns_empty(self):
        assert _extract_code("") == ""

    def test_multiline_code_block(self):
        text = "```python\nimport os\nresult = 1 + 1\n```"
        assert _extract_code(text) == "import os\nresult = 1 + 1"

    def test_strips_surrounding_whitespace(self):
        text = "```python\n\n  result = 7  \n\n```"
        assert _extract_code(text) == "result = 7"


# ═══════════════════════════════════════════════════════════════════════════════
# _execute
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecute:
    def test_string_result(self, small_reg, small_logins):
        output, error = _execute("result = '2 users'", small_reg, small_logins)
        assert error is None
        assert output["result"] == "2 users"

    def test_numeric_result(self, small_reg, small_logins):
        output, error = _execute("result = len(registrations)", small_reg, small_logins)
        assert error is None
        assert output["result"] == 2

    def test_dataframe_result(self, small_reg, small_logins):
        code = "result = registrations[['Username', 'Sex']]"
        output, error = _execute(code, small_reg, small_logins)
        assert error is None
        assert isinstance(output["result"], pd.DataFrame)
        assert list(output["result"].columns) == ["Username", "Sex"]

    def test_fig_result(self, small_reg, small_logins):
        code = "fig = px.bar(registrations, x='Username', y='Username')"
        output, error = _execute(code, small_reg, small_logins)
        assert error is None
        assert "fig" in output

    def test_both_result_and_fig(self, small_reg, small_logins):
        code = "result = 'chart below'\nfig = px.bar(registrations, x='Username', y='Username')"
        output, error = _execute(code, small_reg, small_logins)
        assert error is None
        assert "result" in output
        assert "fig" in output

    def test_exception_returns_error_string(self, small_reg, small_logins):
        output, error = _execute("1 / 0", small_reg, small_logins)
        assert output == {}
        assert error is not None
        assert "ZeroDivisionError" in error

    def test_no_output_returns_empty_dict(self, small_reg, small_logins):
        output, error = _execute("x = 1 + 1", small_reg, small_logins)
        assert output == {}
        assert error is None

    def test_does_not_mutate_originals(self, small_reg, small_logins):
        code = "registrations['new_col'] = 999"
        _execute(code, small_reg, small_logins)
        assert "new_col" not in small_reg.columns

    def test_can_use_pd_and_px(self, small_reg, small_logins):
        code = "df = pd.DataFrame({'a': [1, 2]})\nresult = len(df)"
        output, error = _execute(code, small_reg, small_logins)
        assert error is None
        assert output["result"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# _to_display_blocks
# ═══════════════════════════════════════════════════════════════════════════════


class TestToDisplayBlocks:
    def test_empty_output(self):
        text, blocks = _to_display_blocks({})
        assert text == ""
        assert blocks == []

    def test_string_result_makes_text_block(self):
        text, blocks = _to_display_blocks({"result": "42 people"})
        assert text == "42 people"
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "42 people"

    def test_dataframe_result_makes_dataframe_block(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        text, blocks = _to_display_blocks({"result": df})
        assert text == ""
        assert len(blocks) == 1
        assert blocks[0]["type"] == "dataframe"
        assert blocks[0]["data"].equals(df)

    def test_fig_makes_chart_block_with_valid_json(self):
        fig = px.bar(pd.DataFrame({"x": [1], "y": [2]}), x="x", y="y")
        text, blocks = _to_display_blocks({"fig": fig})
        assert text == ""
        assert len(blocks) == 1
        b = blocks[0]
        assert b["type"] == "chart"
        assert os.path.exists(b["path"])
        with open(b["path"]) as f:
            chart_json = json.load(f)
        assert "data" in chart_json

    def test_both_result_and_fig_makes_two_blocks(self):
        fig = px.scatter(pd.DataFrame({"x": [1], "y": [1]}), x="x", y="y")
        text, blocks = _to_display_blocks({"result": "see chart", "fig": fig})
        assert text == "see chart"
        assert len(blocks) == 2
        types = {b["type"] for b in blocks}
        assert "text" in types
        assert "chart" in types

    def test_empty_dataframe_falls_through_to_text_block(self):
        empty_df = pd.DataFrame()
        text, blocks = _to_display_blocks({"result": empty_df})
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
