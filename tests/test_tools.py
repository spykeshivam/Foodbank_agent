"""
Unit tests for tools.py — all 10 tool functions.

Run with:
    py -3.12 -m pytest tests/ -v
"""
import json
import os
import sys

import pandas as pd
import pytest

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import _call_tool
from tools import (
    _apply_filters,
    _datasets,
    clarify_question,
    create_bar_chart,
    create_line_chart,
    create_pie_chart,
    filter_logins,
    filter_registrations,
    get_column_values,
    group_and_count,
    init_datasets,
    join_sheets,
    summarise_dataframe,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="session", autouse=True)
def loaded_data():
    """Load real Excel files once for the whole test session."""
    reg = pd.read_excel(os.path.join(DATA_DIR, "Registration Form (Responses).xlsx"))
    logins = pd.read_excel(os.path.join(DATA_DIR, "Log In (Responses).xlsx"))
    init_datasets(reg, logins)
    return reg, logins


# ═══════════════════════════════════════════════════════════════════════════════
# 1. clarify_question
# ═══════════════════════════════════════════════════════════════════════════════

class TestClarifyQuestion:
    def test_returns_json_string(self):
        result = clarify_question("How many months back?")
        assert isinstance(result, str)

    def test_clarification_needed_flag(self):
        data = json.loads(clarify_question("How many months back?"))
        assert data["clarification_needed"] is True

    def test_question_echoed(self):
        q = "What dietary requirement are you interested in?"
        data = json.loads(clarify_question(q))
        assert data["question"] == q

    def test_empty_question_still_valid(self):
        data = json.loads(clarify_question(""))
        assert "clarification_needed" in data


# ═══════════════════════════════════════════════════════════════════════════════
# 2. filter_registrations
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilterRegistrations:
    def test_no_filter_returns_all_rows(self):
        data = json.loads(filter_registrations({}))
        assert data["row_count"] == 109

    def test_filter_female(self):
        data = json.loads(filter_registrations({"Sex": "Female"}))
        assert data["row_count"] == 51

    def test_filter_male(self):
        data = json.loads(filter_registrations({"Sex": "Male"}))
        assert data["row_count"] == 58

    def test_filter_halal_substring(self):
        # substring match — catches "Halal", "Halal, No Pork", etc.
        data = json.loads(filter_registrations({"Dietary Requirements": "Halal"}))
        assert data["row_count"] == 45

    def test_filter_female_and_halal(self):
        data = json.loads(filter_registrations({"Sex": "Female", "Dietary Requirements": "Halal"}))
        assert data["row_count"] == 27

    def test_filter_case_insensitive(self):
        lower = json.loads(filter_registrations({"Sex": "female"}))
        upper = json.loads(filter_registrations({"Sex": "Female"}))
        assert lower["row_count"] == upper["row_count"]

    def test_result_stored_as_filtered_registrations(self):
        filter_registrations({"Sex": "Female"})
        assert "filtered_registrations" in _datasets
        assert len(_datasets["filtered_registrations"]) == 51

    def test_first_5_rows_present(self):
        data = json.loads(filter_registrations({}))
        assert len(data["first_5_rows"]) == 5

    def test_unknown_column_ignored_gracefully(self):
        # Should not raise — unknown column is silently skipped
        data = json.loads(filter_registrations({"NonExistentColumn": "value"}))
        assert data["row_count"] == 109

    def test_filter_no_match_returns_zero(self):
        data = json.loads(filter_registrations({"Sex": "Nonbinary"}))
        assert data["row_count"] == 0

    def test_columns_list_in_response(self):
        data = json.loads(filter_registrations({}))
        assert "Username" in data["columns"]
        assert "Sex" in data["columns"]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. filter_logins
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilterLogins:
    def test_no_filter_returns_all_rows(self):
        data = json.loads(filter_logins({}))
        assert data["row_count"] == 1121

    def test_filter_by_day_tuesday(self):
        data = json.loads(filter_logins({"Day": "Tuesday"}))
        assert data["row_count"] == 524

    def test_filter_by_day_friday(self):
        data = json.loads(filter_logins({"Day": "Friday"}))
        assert data["row_count"] == 588

    def test_months_back_beyond_data_returns_zero(self):
        # Data is from 2025; today is 2026 — 3 months back hits empty range
        data = json.loads(filter_logins({}, months_back=3))
        assert data["row_count"] == 0

    def test_months_back_covers_all_data(self):
        # 16 months back from Apr 2026 safely covers all data (Jan–Apr 2025)
        data = json.loads(filter_logins({}, months_back=16))
        assert data["row_count"] == 1121

    def test_months_back_partial(self):
        all_rows = json.loads(filter_logins({}))["row_count"]
        partial = json.loads(filter_logins({}, months_back=14))["row_count"]
        assert 0 < partial < all_rows

    def test_result_stored_as_filtered_logins(self):
        filter_logins({"Day": "Friday"})
        assert "filtered_logins" in _datasets

    def test_unknown_column_ignored(self):
        data = json.loads(filter_logins({"BadCol": "x"}))
        assert data["row_count"] == 1121


# ═══════════════════════════════════════════════════════════════════════════════
# 4. join_sheets
# ═══════════════════════════════════════════════════════════════════════════════

class TestJoinSheets:
    def test_basic_join_row_count(self):
        data = json.loads(join_sheets({}, {}))
        assert data["row_count"] == 296

    def test_join_female_halal(self):
        data = json.loads(join_sheets({}, {"Sex": "Female", "Dietary Requirements": "Halal"}))
        assert data["row_count"] == 72

    def test_join_tuesday_only(self):
        data = json.loads(join_sheets({"Day": "Tuesday"}, {}))
        assert data["row_count"] > 0

    def test_months_back_zero_data_range(self):
        data = json.loads(join_sheets({}, {}, months_back=3))
        assert data["row_count"] == 0

    def test_months_back_full_coverage(self):
        data = json.loads(join_sheets({}, {}, months_back=16))
        assert data["row_count"] == 296

    def test_joined_dataset_stored(self):
        join_sheets({}, {})
        assert "joined" in _datasets

    def test_joined_has_timestamp_column(self):
        join_sheets({}, {})
        assert "Timestamp" in _datasets["joined"].columns

    def test_joined_contains_both_sheet_columns(self):
        join_sheets({}, {})
        cols = list(_datasets["joined"].columns)
        # From logins
        assert "Day" in cols
        # From registrations
        assert "Sex" in cols

    def test_no_cross_join_only_matching_usernames(self):
        # 76 usernames appear in both sheets
        join_sheets({}, {})
        n_unique = _datasets["joined"]["Username"].nunique()
        assert n_unique <= 76


# ═══════════════════════════════════════════════════════════════════════════════
# 5. group_and_count
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroupAndCount:
    def test_group_registrations_by_sex(self):
        data = json.loads(group_and_count("registrations", ["Sex"]))
        rows = {r["Sex"]: r["count"] for r in data["data"]}
        assert rows["Female"] == 51
        assert rows["Male"] == 58

    def test_group_logins_by_day(self):
        data = json.loads(group_and_count("logins", ["Day"]))
        rows = {r["Day"]: r["count"] for r in data["data"]}
        assert rows["Tuesday"] == 524
        assert rows["Friday"] == 588

    def test_group_logins_by_month(self):
        data = json.loads(group_and_count("logins", ["month"]))
        rows = {r["Month"]: r["count"] for r in data["data"]}
        assert rows["2025-01"] == 100
        assert rows["2025-02"] == 451
        assert rows["2025-03"] == 442
        assert rows["2025-04"] == 128

    def test_total_count_preserved(self):
        data = json.loads(group_and_count("logins", ["Day"]))
        total = sum(r["count"] for r in data["data"])
        assert total == 1121

    def test_dataset_key_in_response(self):
        data = json.loads(group_and_count("registrations", ["Sex"]))
        assert "dataset_key" in data
        assert data["dataset_key"] in _datasets

    def test_unknown_column_returns_error(self):
        data = json.loads(group_and_count("registrations", ["NonExistent"]))
        assert "error" in data

    def test_invalid_dataset_raises(self):
        with pytest.raises(KeyError):
            group_and_count("nonexistent_dataset", ["Sex"])

    def test_months_back_filters_before_grouping(self):
        # 14 months back from today excludes the earliest months in the data
        all_data = json.loads(group_and_count("logins", ["month"]))
        partial_data = json.loads(group_and_count("logins", ["month"], months_back=14))
        all_total = sum(r["count"] for r in all_data["data"])
        partial_total = sum(r["count"] for r in partial_data["data"])
        assert 0 < partial_total < all_total
        assert partial_data["row_count"] < all_data["row_count"]

    def test_group_by_multiple_columns(self):
        join_sheets({}, {})
        data = json.loads(group_and_count("joined", ["Sex", "Day"]))
        assert "error" not in data
        assert data["row_count"] > 0

    def test_row_count_matches_data_length(self):
        data = json.loads(group_and_count("registrations", ["Sex"]))
        assert data["row_count"] == len(data["data"])


# ═══════════════════════════════════════════════════════════════════════════════
# 6–8. Chart tools
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_DATA = json.dumps([
    {"Label": "A", "count": 10},
    {"Label": "B", "count": 25},
    {"Label": "C", "count": 5},
])

SAMPLE_TIME_DATA = json.dumps([
    {"Month": "2025-01", "count": 100},
    {"Month": "2025-02", "count": 451},
    {"Month": "2025-03", "count": 442},
])


class TestCreateBarChart:
    def test_returns_chart_path(self):
        result = json.loads(create_bar_chart(SAMPLE_DATA, "Label", "count", None, "Test Bar", "group"))
        assert "chart_path" in result

    def test_chart_file_exists(self):
        result = json.loads(create_bar_chart(SAMPLE_DATA, "Label", "count", None, "Test Bar", "group"))
        assert os.path.exists(result["chart_path"])

    def test_chart_file_is_valid_json(self):
        result = json.loads(create_bar_chart(SAMPLE_DATA, "Label", "count", None, "Test Bar", "group"))
        with open(result["chart_path"]) as f:
            chart_data = json.load(f)
        assert "data" in chart_data

    def test_stack_barmode(self):
        result = json.loads(create_bar_chart(SAMPLE_DATA, "Label", "count", None, "Stacked", "stack"))
        assert os.path.exists(result["chart_path"])

    def test_with_color_column(self):
        data = json.dumps([
            {"Label": "A", "Group": "X", "count": 10},
            {"Label": "B", "Group": "Y", "count": 20},
        ])
        result = json.loads(create_bar_chart(data, "Label", "count", "Group", "Grouped", "group"))
        assert os.path.exists(result["chart_path"])

    def test_invalid_column_raises(self):
        with pytest.raises(Exception):
            create_bar_chart(SAMPLE_DATA, "BadX", "BadY", None, "Bad", "group")


class TestCreateLineChart:
    def test_returns_chart_path(self):
        result = json.loads(create_line_chart(SAMPLE_TIME_DATA, "Month", "count", None, "Trend"))
        assert "chart_path" in result

    def test_chart_file_valid(self):
        result = json.loads(create_line_chart(SAMPLE_TIME_DATA, "Month", "count", None, "Trend"))
        with open(result["chart_path"]) as f:
            chart_data = json.load(f)
        assert "data" in chart_data

    def test_with_color(self):
        data = json.dumps([
            {"Month": "2025-01", "Sex": "Female", "count": 40},
            {"Month": "2025-01", "Sex": "Male", "count": 60},
            {"Month": "2025-02", "Sex": "Female", "count": 45},
            {"Month": "2025-02", "Sex": "Male", "count": 55},
        ])
        result = json.loads(create_line_chart(data, "Month", "count", "Sex", "By Sex"))
        assert os.path.exists(result["chart_path"])


class TestCreatePieChart:
    def test_returns_chart_path(self):
        result = json.loads(create_pie_chart(SAMPLE_DATA, "Label", "count", "Pie Test"))
        assert "chart_path" in result

    def test_chart_file_valid(self):
        result = json.loads(create_pie_chart(SAMPLE_DATA, "Label", "count", "Pie Test"))
        with open(result["chart_path"]) as f:
            chart_data = json.load(f)
        assert "data" in chart_data

    def test_sex_breakdown_pie(self):
        grouped = json.loads(group_and_count("registrations", ["Sex"]))
        data_json = json.dumps(grouped["data"])
        result = json.loads(create_pie_chart(data_json, "Sex", "count", "Sex Breakdown"))
        assert os.path.exists(result["chart_path"])


# ═══════════════════════════════════════════════════════════════════════════════
# 9. summarise_dataframe
# ═══════════════════════════════════════════════════════════════════════════════

class TestSummariseDataframe:
    def setup_method(self):
        # Store custom DataFrames under private test keys in _datasets
        _datasets["_test_numeric"] = pd.DataFrame([
            {"Adults": 1, "Children": 0},
            {"Adults": 2, "Children": 1},
            {"Adults": 3, "Children": 2},
            {"Adults": 1, "Children": 0},
        ])
        _datasets["_test_cat"] = pd.DataFrame([
            {"Sex": "Female", "Lang": "Bengali"},
            {"Sex": "Male",   "Lang": "English"},
            {"Sex": "Female", "Lang": "Bengali"},
            {"Sex": "Male",   "Lang": "Arabic"},
            {"Sex": "Female", "Lang": "Bengali"},
        ])

    def test_numeric_column_returns_stats(self):
        result = json.loads(summarise_dataframe("_test_numeric", ["Adults"]))
        stats = result["Adults"]
        assert stats["count"] == 4
        assert stats["mean"] == 1.75
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0
        assert stats["sum"] == 7.0

    def test_categorical_column_returns_top_values(self):
        result = json.loads(summarise_dataframe("_test_cat", ["Sex"]))
        stats = result["Sex"]
        assert stats["count"] == 5
        assert stats["unique"] == 2
        assert "Female" in stats["top_values"]

    def test_multiple_columns(self):
        result = json.loads(summarise_dataframe("_test_cat", ["Sex", "Lang"]))
        assert "Sex" in result
        assert "Lang" in result

    def test_missing_column_returns_not_found(self):
        result = json.loads(summarise_dataframe("_test_numeric", ["NonExistent"]))
        assert result["NonExistent"] == "column not found"

    def test_real_data_registrations(self):
        result = json.loads(summarise_dataframe("registrations", ["Number of Adults in Household"]))
        assert "Number of Adults in Household" in result

    def test_filtered_dataset(self):
        filter_registrations({"Sex": "Female"})  # stores filtered_registrations
        result = json.loads(summarise_dataframe("filtered_registrations", ["Number of Adults in Household"]))
        assert "Number of Adults in Household" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 10. get_column_values
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetColumnValues:
    def test_sex_values(self):
        result = json.loads(get_column_values("registrations", "Sex"))
        assert set(result["unique_values"]) == {"Male", "Female"}

    def test_day_values(self):
        result = json.loads(get_column_values("logins", "Day"))
        assert set(result["unique_values"]) == {"Tuesday", "Friday", "Wednesday"}

    def test_unknown_column_returns_error(self):
        result = json.loads(get_column_values("registrations", "NotAColumn"))
        assert "error" in result

    def test_unknown_sheet_raises(self):
        with pytest.raises(KeyError):
            get_column_values("nonexistent_sheet", "Sex")

    def test_capped_at_100_values(self):
        # Username has 109 unique values — should be capped at 100
        result = json.loads(get_column_values("registrations", "Username"))
        assert len(result["unique_values"]) <= 100

    def test_response_includes_sheet_and_column(self):
        result = json.loads(get_column_values("registrations", "Sex"))
        assert result["sheet"] == "registrations"
        assert result["column"] == "Sex"

    def test_dietary_requirements_includes_halal(self):
        result = json.loads(get_column_values("registrations", "Dietary Requirements"))
        values_str = " ".join(str(v) for v in result["unique_values"])
        assert "Halal" in values_str


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyFilters:
    def setup_method(self):
        self.df = pd.DataFrame({
            "Name": ["Alice", "Bob", "Charlie", "alice"],
            "Score": [10, 20, 30, 40],
            "Tag": ["foo", "bar", "foo", "baz"],
        })

    def test_string_substring_match(self):
        result = _apply_filters(self.df.copy(), {"Name": "alice"})
        assert len(result) == 2  # "Alice" and "alice"

    def test_numeric_exact_match(self):
        result = _apply_filters(self.df.copy(), {"Score": 20})
        assert len(result) == 1

    def test_multiple_filters_anded(self):
        result = _apply_filters(self.df.copy(), {"Tag": "foo", "Score": 10})
        assert len(result) == 1

    def test_unknown_column_ignored(self):
        result = _apply_filters(self.df.copy(), {"BadCol": "x"})
        assert len(result) == 4

    def test_no_match_empty_result(self):
        result = _apply_filters(self.df.copy(), {"Name": "Zara"})
        assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: tool chaining
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolChaining:
    """Simulate the multi-step flows the LLM agent uses."""

    def test_flow_sex_breakdown_pie_chart(self):
        """group_and_count → create_pie_chart"""
        grouped = json.loads(group_and_count("registrations", ["Sex"]))
        data_json = json.dumps(grouped["data"])
        chart = json.loads(create_pie_chart(data_json, "Sex", "count", "Sex Breakdown"))
        assert os.path.exists(chart["chart_path"])

    def test_flow_monthly_logins_bar_chart(self):
        """group_and_count(month) → create_bar_chart"""
        grouped = json.loads(group_and_count("logins", ["month"]))
        data_json = json.dumps(grouped["data"])
        chart = json.loads(create_bar_chart(data_json, "Month", "count", None, "Monthly Logins", "group"))
        assert os.path.exists(chart["chart_path"])

    def test_flow_join_then_group_by_day(self):
        """join_sheets → group_and_count(Day)"""
        join_result = json.loads(join_sheets({}, {"Sex": "Female"}))
        assert join_result["row_count"] > 0
        grouped = json.loads(group_and_count("joined", ["Day"]))
        total = sum(r["count"] for r in grouped["data"])
        assert total == join_result["row_count"]

    def test_flow_filter_then_summarise(self):
        """filter_registrations → summarise_dataframe (pass dataset name, not raw JSON)"""
        filter_registrations({"Sex": "Female"})  # stores "filtered_registrations"
        summary = json.loads(summarise_dataframe("filtered_registrations", ["Number of Adults in Household"]))
        assert "Number of Adults in Household" in summary

    def test_flow_get_values_then_filter(self):
        """get_column_values → filter_registrations using a discovered value"""
        vals = json.loads(get_column_values("registrations", "Sex"))["unique_values"]
        assert len(vals) > 0
        first_val = vals[0]
        result = json.loads(filter_registrations({"Sex": first_val}))
        assert result["row_count"] > 0

    def test_flow_full_halal_female_logins(self):
        """join_sheets → group_and_count → create_bar_chart (the example from spec)"""
        join_result = json.loads(join_sheets(
            {},
            {"Sex": "Female", "Dietary Requirements": "Halal"},
            months_back=None,
        ))
        assert join_result["row_count"] == 72

        grouped = json.loads(group_and_count("joined", ["month"]))
        assert grouped["row_count"] > 0
        total = sum(r["count"] for r in grouped["data"])
        assert total == 72

        data_json = json.dumps(grouped["data"])
        chart = json.loads(create_bar_chart(
            data_json, "Month", "count", None,
            "Female Halal Logins per Month", "group"
        ))
        assert os.path.exists(chart["chart_path"])

    def test_flow_line_chart_trend(self):
        """group_and_count(month) → create_line_chart"""
        grouped = json.loads(group_and_count("logins", ["month"]))
        data_json = json.dumps(grouped["data"])
        chart = json.loads(create_line_chart(data_json, "Month", "count", None, "Login Trend"))
        assert os.path.exists(chart["chart_path"])


# ═══════════════════════════════════════════════════════════════════════════════
# Defensive: bad tool calls must return JSON errors, never crash the agent
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolDefensiveness:
    """
    Reproduce the exact mistakes the model made in production and assert that
    _call_tool always returns a JSON error string instead of raising.
    """

    def test_unknown_tool_returns_json_error(self):
        result = json.loads(_call_tool("nonexistent_tool", {}))
        assert "error" in result

    def test_group_and_count_with_filters_kwarg_returns_json_error(self):
        """Reproduces: group_and_count() got an unexpected keyword argument 'filters'"""
        result = json.loads(_call_tool("group_and_count", {
            "dataset": "registrations",
            "group_by": ["Sex"],
            "filters": {"Sex": "Female"},   # does not exist in signature
        }))
        assert "error" in result

    def test_group_and_count_with_filters_kwarg_does_not_raise(self):
        """Same as above but verifies no exception escapes _call_tool."""
        try:
            _call_tool("group_and_count", {
                "dataset": "registrations",
                "group_by": ["Sex"],
                "filters": {"Sex": "Female"},
            })
        except Exception as exc:
            pytest.fail(f"_call_tool raised instead of returning JSON error: {exc}")

    def test_summarise_dataframe_with_raw_json_returns_json_error(self):
        """Reproduces: model passes '[{...}]' as dataset instead of a dataset name."""
        result = json.loads(_call_tool("summarise_dataframe", {
            "dataset": '[{"Adults": 1, "Children": 0}]',
            "columns": ["Adults"],
        }))
        assert "error" in result

    def test_summarise_dataframe_with_old_data_json_param_returns_json_error(self):
        """Reproduces: model uses old param name data_json= after API change."""
        result = json.loads(_call_tool("summarise_dataframe", {
            "data_json": '[{"Adults": 1}]',
            "columns": ["Adults"],
        }))
        assert "error" in result

    def test_summarise_dataframe_unknown_dataset_returns_json_error(self):
        result = json.loads(_call_tool("summarise_dataframe", {
            "dataset": "does_not_exist",
            "columns": ["Sex"],
        }))
        assert "error" in result

    def test_group_and_count_unknown_dataset_returns_json_error(self):
        result = json.loads(_call_tool("group_and_count", {
            "dataset": "does_not_exist",
            "group_by": ["Sex"],
        }))
        assert "error" in result

    def test_all_tool_errors_are_valid_json(self):
        """Every error path must return parseable JSON — the agent depends on json.loads()."""
        bad_calls = [
            ("nonexistent_tool", {}),
            ("group_and_count", {"dataset": "x", "group_by": ["y"], "filters": {}}),
            ("summarise_dataframe", {"dataset": '[{"x":1}]', "columns": ["x"]}),
            ("summarise_dataframe", {"data_json": "[]", "columns": []}),
            ("filter_registrations", {"filters": {"NonExistentCol": "val"}}),
        ]
        for tool_name, args in bad_calls:
            raw = _call_tool(tool_name, args)
            parsed = json.loads(raw)   # must not raise
            assert isinstance(parsed, dict), f"{tool_name}: expected dict, got {type(parsed)}"
