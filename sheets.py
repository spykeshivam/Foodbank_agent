"""Google Sheets data loader using gspread + service account credentials."""
import os

import gspread

from config import CREDENTIALS_FILE

_SHEET_ID = os.environ["SHEET_ID"]
_LOGIN_SHEET_ID = os.environ["LOGIN_SHEET_ID"]


def _client() -> gspread.Client:
    return gspread.service_account(filename=CREDENTIALS_FILE)


def fetch_sheet_data(sheet_id: str, worksheet_name: str) -> list[dict]:
    """
    Open *worksheet_name* inside the Google Spreadsheet identified by
    *sheet_id* and return all rows as a list of dicts keyed by the
    header row.
    """
    gc = _client()
    spreadsheet = gc.open_by_key(sheet_id)
    worksheet = spreadsheet.worksheet(worksheet_name)
    return worksheet.get_all_records()


def fetch_registrations() -> list[dict]:
    return fetch_sheet_data(_SHEET_ID, "Form Responses 1")


def fetch_logins() -> list[dict]:
    return fetch_sheet_data(_LOGIN_SHEET_ID, "Form Responses 1")
