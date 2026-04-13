"""
Integration tests for sheets.py — verifies real data can be loaded
from Google Sheets using the service account credentials.

Requires:
  - credentials.json present in the project root
  - Network access to Google Sheets API

Run with:
    uv run pytest tests/test_sheets.py -v
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sheets import fetch_registrations, fetch_logins

# ── Skip entire module if credentials file is missing ─────────────────────────
from config import CREDENTIALS_FILE

pytestmark = pytest.mark.skipif(
    not os.path.exists(CREDENTIALS_FILE),
    reason=f"credentials.json not found at {CREDENTIALS_FILE} — skipping sheet tests",
)

# ── Fixtures — fetch once per session ─────────────────────────────────────────

@pytest.fixture(scope="session")
def registrations() -> list[dict]:
    return fetch_registrations()


@pytest.fixture(scope="session")
def logins() -> list[dict]:
    return fetch_logins()


@pytest.fixture(scope="session")
def reg_df(registrations) -> pd.DataFrame:
    return pd.DataFrame(registrations)


@pytest.fixture(scope="session")
def login_df(logins) -> pd.DataFrame:
    return pd.DataFrame(logins)


# ═══════════════════════════════════════════════════════════════════════════════
# Registrations sheet
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegistrationsLoad:
    def test_returns_list(self, registrations):
        assert isinstance(registrations, list)

    def test_not_empty(self, registrations):
        assert len(registrations) > 0, "Registrations sheet returned 0 rows"

    def test_each_row_is_dict(self, registrations):
        for row in registrations[:5]:
            assert isinstance(row, dict)

    def test_no_completely_null_rows(self, registrations):
        for i, row in enumerate(registrations):
            assert any(v not in (None, "", "N/A") for v in row.values()), \
                f"Row {i} is entirely null/empty"

    def test_expected_columns_present(self, reg_df):
        required = [
            "Username",
            "First Name",
            "Surname",
            "Sex",
            "Dietary Requirements",
            "Primary Spoken Language",
            "Number of Adults in Household",
            "Number of Children in Household",
        ]
        missing = [c for c in required if c not in reg_df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_username_column_not_null(self, reg_df):
        null_count = reg_df["Username"].isnull().sum()
        assert null_count == 0, f"{null_count} rows have a null Username"

    def test_username_values_are_strings(self, reg_df):
        assert pd.api.types.is_string_dtype(reg_df["Username"])

    def test_username_no_duplicates(self, reg_df):
        dupes = reg_df["Username"].duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate Usernames found"

    def test_sex_only_known_values(self, reg_df):
        allowed = {"Male", "Female", ""}
        actual = set(reg_df["Sex"].fillna("").unique())
        unexpected = actual - allowed
        assert not unexpected, f"Unexpected Sex values: {unexpected}"

    def test_adults_in_household_is_numeric(self, reg_df):
        col = pd.to_numeric(reg_df["Number of Adults in Household"], errors="coerce")
        null_count = col.isnull().sum()
        assert null_count == 0, f"{null_count} non-numeric values in 'Number of Adults in Household'"

    def test_adults_in_household_positive(self, reg_df):
        col = pd.to_numeric(reg_df["Number of Adults in Household"], errors="coerce")
        assert (col >= 0).all(), "Found negative values in 'Number of Adults in Household'"

    def test_dataframe_has_rows_and_columns(self, reg_df):
        assert reg_df.shape[0] > 0
        assert reg_df.shape[1] > 5


# ═══════════════════════════════════════════════════════════════════════════════
# Logins sheet
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoginsLoad:
    def test_returns_list(self, logins):
        assert isinstance(logins, list)

    def test_not_empty(self, logins):
        assert len(logins) > 0, "Logins sheet returned 0 rows"

    def test_each_row_is_dict(self, logins):
        for row in logins[:5]:
            assert isinstance(row, dict)

    def test_no_completely_null_rows(self, logins):
        for i, row in enumerate(logins):
            assert any(v not in (None, "", "N/A") for v in row.values()), \
                f"Row {i} is entirely null/empty"

    def test_expected_columns_present(self, login_df):
        required = ["Username", "Timestamp", "Day"]
        missing = [c for c in required if c not in login_df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_username_column_not_null(self, login_df):
        null_count = login_df["Username"].isnull().sum()
        assert null_count == 0, f"{null_count} rows have a null Username"

    def test_timestamp_column_not_null(self, login_df):
        null_count = login_df["Timestamp"].isnull().sum()
        assert null_count == 0, f"{null_count} rows have a null Timestamp"

    def test_timestamp_is_parseable(self, login_df):
        parsed = pd.to_datetime(login_df["Timestamp"], format="mixed", errors="coerce")
        null_count = parsed.isnull().sum()
        assert null_count == 0, f"{null_count} Timestamp values could not be parsed as dates"

    def test_day_only_known_values(self, login_df):
        allowed = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        actual = set(login_df["Day"].fillna("").unique()) - {""}
        unexpected = actual - allowed
        assert not unexpected, f"Unexpected Day values: {unexpected}"

    def test_dataframe_has_rows_and_columns(self, login_df):
        assert login_df.shape[0] > 0
        assert login_df.shape[1] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-sheet integrity
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossSheetIntegrity:
    def test_login_usernames_exist_in_registrations(self, reg_df, login_df):
        reg_usernames = set(reg_df["Username"])
        login_usernames = set(login_df["Username"])
        orphans = login_usernames - reg_usernames
        # Warn if >10% of login usernames have no registration record
        pct_orphan = len(orphans) / len(login_usernames) * 100
        assert pct_orphan < 10, (
            f"{len(orphans)} login usernames ({pct_orphan:.1f}%) "
            f"have no matching registration record"
        )

    def test_join_produces_rows(self, reg_df, login_df):
        joined = login_df.merge(reg_df, on="Username", how="inner")
        assert len(joined) > 0, "Inner join on Username produced 0 rows"
