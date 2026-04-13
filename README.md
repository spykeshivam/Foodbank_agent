# Foodbank Admin AI

An interactive AI chat interface for St Dunstan's Food Bank admins.
Ask questions about registrations and visit data in plain English — get back
numerical answers, tables, and charts.

---

## What it does

- Natural-language queries against two Excel datasets (registrations + logins)
- Powered by Google Gemini 1.5 Flash (free tier, no credit card needed)
- Responses include plain text, pandas DataFrames rendered as tables, and interactive Plotly charts
- Multi-step agentic loop — the AI chains multiple tool calls to answer complex questions
- Asks clarifying questions when a query is ambiguous (e.g. "past few months" → asks how many)

---

## Project structure

```
Foodbank_agent/
├── app.py              # Streamlit UI + agentic loop
├── tools.py            # All data tool implementations
├── tool_schemas.py     # Gemini function-calling schemas
├── requirements.txt    # Python dependencies
├── tests/
│   └── test_tools.py   # 78 unit + integration tests
└── data/               # NOT in git — see Setup below
    ├── Registration Form (Responses).xlsx
    └── Log In (Responses).xlsx
```

---

## Setup (step by step)

### 1. Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager
- Git — download from https://git-scm.com

Install uv (one-time):
```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone https://github.com/spykeshivam/Foodbank_agent.git
cd Foodbank_agent
```

### 3. Install dependencies and run

uv handles the virtual environment automatically — no manual `venv` step needed:

```bash
uv sync
```

To run the app:
```bash
uv run streamlit run app.py
```

To run the tests:
```bash
uv run pytest tests/ -v
```

### 5. Get a Gemini API key (free)

1. Go to **https://aistudio.google.com**
2. Sign in with a Google account
3. Click **"Get API key"** in the left sidebar
4. Click **"Create API key"** → copy it

Set it as a permanent environment variable:

Windows PowerShell (run once — survives reboots):
```powershell
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "YOUR_KEY_HERE", "User")
```

Then **open a new terminal** and verify:
```powershell
echo $env:GEMINI_API_KEY
```

macOS / Linux (`~/.zshrc` or `~/.bashrc`):
```bash
export GEMINI_API_KEY="YOUR_KEY_HERE"
```

### 6. Add the data files

The Excel files are **not stored in git** (they contain personal data).
Obtain them from the project lead and place them here:

```
data/Registration Form (Responses).xlsx
data/Log In (Responses).xlsx
```

Create the folder if needed:
```bash
mkdir data
```

### 7. Run the app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## Running the tests

```bash
python -m pytest tests/ -v
```

Expected output: **78 passed**

---

## Example questions to try

| Question | What the agent does |
|---|---|
| How many people have Halal dietary requirements? | Filters registrations, returns count |
| Show me the gender breakdown as a pie chart. | Groups by Sex, renders pie chart |
| How many logins were there each month? | Groups logins by month, renders bar chart |
| What are the top 5 spoken languages? | Groups by Primary Spoken Language, shows table + bar chart |
| How many female users with Halal requirements visited in the past few months? | Asks how many months → joins sheets → groups by month → bar chart |

---

## Tech stack

| Component | Library |
|---|---|
| UI | Streamlit |
| AI model | Google Gemini 1.5 Flash (`google-genai`) |
| Data processing | pandas |
| Charts | Plotly |
| Excel reading | openpyxl |
| Tests | pytest |
| Package manager | uv |

---

## Troubleshooting

**`KeyError: GEMINI_API_KEY`**
→ The environment variable is not set, or you haven't opened a new terminal since setting it.

**`FileNotFoundError` on the Excel files**
→ Make sure both files are inside a `data/` folder in the project root with the exact filenames listed above.

**`ModuleNotFoundError: No module named 'google.genai'`**
→ Run `uv sync` to install all dependencies.

**Pylance shows "Import google.genai could not be resolved" in VS Code**
→ Press `Ctrl+Shift+P` → "Python: Select Interpreter" → choose the Python from your `.venv` folder.
