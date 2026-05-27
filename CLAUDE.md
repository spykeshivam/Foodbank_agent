# Foodbank Admin AI — Claude Code Guide

## What this project is

A Streamlit chat app that lets staff at **St Dunstan's Food Bank** query their operational data in plain English. An AI agent (Llama 3.3 70B) answers questions about registrations and visit logs, producing text, tables, and Plotly charts.

Deployed at: https://foodbank-agent.onrender.com (Render free tier — cold starts take ~50s after inactivity).

---

## Architecture

```
User (browser)
    ↓
app.py          — Streamlit UI, auth gate, session state, rendering
    ↓
agent.py        — agentic loop (OpenAI-compatible API, tool calling)
    ↓
tools.py        — 10 tool functions the model can call
    ↓
sheets.py       — reads from two Google Sheets via gspread
```

Supporting files:
- `tool_schemas.py` — JSON Schema definitions for all 10 tools (passed to the API)
- `config.py` — Google credentials setup (decodes `GOOGLE_CREDENTIALS_B64` env var to a temp file)
- `log_config.py` — file logger capped at 3000 lines, writes to `foodbank_agent.log`
- `render.yaml` — Render deployment config

---

## Running locally

```bash
# Install dependencies (requires Python 3.11+)
pip install uv
uv sync

# Set required env vars (copy from Render dashboard)
export GROQ_API_KEY=...
export CEREBRAS_API_KEY=...
export GOOGLE_CREDENTIALS_B64=...   # base64-encoded contents of credentials.json
export SHEET_ID=...                 # registrations Google Sheet ID
export LOGIN_SHEET_ID=...           # logins Google Sheet ID
export APP_USERNAME=...
export APP_PASSWORD=...

# Run
uv run streamlit run app.py
```

Alternatively, place `credentials.json` in the project root and set `GOOGLE_APPLICATION_CREDENTIALS=credentials.json` instead of `GOOGLE_CREDENTIALS_B64`.

---

## LLM providers

Both providers serve **Llama 3.3 70B** via an OpenAI-compatible API. The user picks one from the sidebar dropdown — conversation history is portable between them (same message format).

| Provider | Model name | Base URL | API key env var |
|---|---|---|---|
| Groq | `llama-3.3-70b-versatile` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| Cerebras | `llama3.3-70b` | `https://api.cerebras.ai/v1` | `CEREBRAS_API_KEY` |

Config lives in `PROVIDERS` dict at the top of `agent.py`. To add a new provider, add an entry there — no other changes needed.

Key setting: `parallel_tool_calls=False` is required. Without it, Llama models on Groq fall back to Hermes XML function-call format and the API returns a 400.

---

## Data sources

Two Google Sheets loaded at startup, cached for 5 minutes (`@st.cache_data(ttl=300)`):

- **Registrations** (`SHEET_ID`, worksheet `"Form Responses 1"`) — one row per registered user. Key columns: `Username`, `First Name`, `Surname`, `Date of Birth`, `Sex`, `Postcode`, `Primary Spoken Language`, `Dietary Requirements`, `Ethnicity`, `Relationship Status`, `Property Type`, `Number of Adults in Household`, `Number of Children in Household`, `Cooking Facilities`, `Timestamp`.
- **Logins** (`LOGIN_SHEET_ID`, worksheet `"Form Responses 1"`) — one row per visit. Columns: `Username`, `Timestamp`, `Day` (day-of-week string, NOT a calendar date).

They join on `Username`. The agent uses `join_sheets` to cross-reference them.

---

## Agentic loop (`agent.py`)

`run_query()` → `_loop()` → model calls tools → tools return JSON → repeat → final answer.

- Up to **10 tool calls** per query (`MAX_TOOL_CALLS`).
- **Clarification pause**: if the model calls `clarify_question`, the loop returns an `AgentResponse` with `clarification_question` set and all resume state stored in `_paused_*` fields. `app.py` stores this in session state and calls `continue_after_clarification()` with the user's answer.
- **Retry logic**: 429 → 40s linear backoff (up to 5 attempts); 503/504 and httpx timeouts → 5s backoff.
- The system message (including today's date) is prepended fresh on every `run_query` call and stripped from `AgentResponse.history` before it's stored in session state.

---

## Tools (`tools.py`)

| Tool | Purpose |
|---|---|
| `clarify_question` | Pause and ask the user to clarify an ambiguous query |
| `filter_registrations` | Filter registrations by column/value (case-insensitive substring) |
| `filter_logins` | Filter logins, optionally within last N months |
| `join_sheets` | Inner-join logins + registrations on Username |
| `group_and_count` | Group a dataset by columns and count rows; supports `"month"` and `"date"` virtual columns derived from Timestamp |
| `create_bar_chart` | Plotly bar chart saved as JSON to a temp file |
| `create_line_chart` | Plotly line chart |
| `create_pie_chart` | Plotly donut chart |
| `summarise_dataframe` | Descriptive stats for specific columns |
| `get_column_values` | List unique values in a column (for valid filter values) |

`TOOL_FUNCTIONS` dict in `tools.py` is the dispatch table. `TOOL_SCHEMAS` in `tool_schemas.py` are the JSON Schema definitions sent to the API. Both must be kept in sync when adding a tool.

Charts are saved as Plotly JSON to `tempfile.gettempdir()` and rendered in `app.py` via `plotly.io.from_json`.

---

## Tests

```bash
uv run pytest tests/test_sdk_api.py -v        # 28 unit tests, no credentials needed — always run these
uv run pytest tests/test_agent_queries.py -v  # needs GROQ_API_KEY + Google Sheets credentials
uv run pytest tests/test_sheets.py -v         # needs Google Sheets credentials
# test_tools.py needs Excel files in data/ directory (not in repo)
```

`test_sdk_api.py` is the most important — it catches SDK API surface changes and verifies the provider config, tool schemas, retry logic, and client construction without any live API calls.

---

## Linting & formatting

```bash
uv run ruff check .          # lint
uv run ruff check . --fix    # lint + auto-fix
uv run ruff format .         # format
```

Rules: `E`, `F`, `I` (isort), `UP` (pyupgrade), `B` (bugbear), `SIM` (simplify). Line length 120. `E501` ignored (decorative `# ── ...` section comments exceed 120 chars intentionally).

---

## CI (GitHub Actions)

Three jobs on every push/PR to `master`:

1. **lint** — `ruff check` + `ruff format --check` (no secrets needed)
2. **sdk-tests** — `test_sdk_api.py` (no secrets needed)
3. **integration-tests** — `test_agent_queries.py` + `test_sheets.py` with real keys, only runs when repo variable `INTEGRATION_TESTS_ENABLED=true`

---

## Key decisions & gotchas

- **`parallel_tool_calls=False`** — mandatory for Llama on Groq, otherwise the model generates Hermes-format tool calls and the API returns 400.
- **`Day` column in logins** is day-of-week (e.g. `"Tuesday"`), not a calendar date. Use `"date"` as a `group_by` value in `group_and_count` for calendar dates.
- **Cerebras model name** is `llama3.3-70b` (no dash between "llama" and "3"), unlike Groq's `llama-3.3-70b-versatile`.
- **History format**: `AgentResponse.history` contains only conversation turn messages (no system message). The system message is rebuilt and prepended fresh on each `run_query` call.
- **Google credentials**: on Render, set `GOOGLE_CREDENTIALS_B64` (base64 of `credentials.json`). `config.py` decodes it to a temp file at import time.
- The Render free tier **spins down after inactivity** — first request after idle takes 50s+. Nothing to fix without upgrading.
