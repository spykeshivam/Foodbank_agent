"""Tool implementations for the Foodbank AI agent."""
import json
import re
import tempfile
from datetime import datetime
from io import StringIO
from dateutil.relativedelta import relativedelta

import pandas as pd
import plotly.express as px

# ── Global state shared with app.py ──────────────────────────────────────────
_datasets: dict[str, pd.DataFrame] = {}


def init_datasets(registrations: pd.DataFrame, logins: pd.DataFrame) -> None:
    """Called once at startup to inject the loaded DataFrames."""
    _datasets["registrations"] = registrations.copy()
    _datasets["logins"] = logins.copy()
    # Ensure logins Timestamp is datetime — data has mixed formats (M/D/YYYY and ISO)
    _datasets["logins"]["Timestamp"] = pd.to_datetime(
        _datasets["logins"]["Timestamp"], format="mixed", dayfirst=False
    )
    # Ensure registrations Timestamp is datetime
    _datasets["registrations"]["Timestamp"] = pd.to_datetime(
        _datasets["registrations"]["Timestamp"], format="mixed", dayfirst=False
    )


def _get(name: str) -> pd.DataFrame:
    if name not in _datasets:
        raise KeyError(f"Dataset '{name}' not found. Available: {list(_datasets.keys())}")
    return _datasets[name].copy()


def _months_back_cutoff(months_back: int | None) -> datetime | None:
    if months_back is None:
        return None
    return datetime.now() - relativedelta(months=months_back)


def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply column=value filters (case-insensitive whole-word match for strings)."""
    for col, val in filters.items():
        if col not in df.columns:
            continue
        if isinstance(val, str):
            pattern = r"(?<![A-Za-z])" + re.escape(val) + r"(?![A-Za-z])"
            df = df[df[col].astype(str).str.contains(pattern, case=False, na=False, regex=True)]
        else:
            df = df[df[col] == val]
    return df


def _df_summary(df: pd.DataFrame, label: str = "") -> str:
    summary = {
        "label": label,
        "row_count": len(df),
        "columns": list(df.columns),
        "first_5_rows": json.loads(df.head(5).to_json(orient="records", date_format="iso")),
    }
    return json.dumps(summary)


# ── 1. clarify_question ───────────────────────────────────────────────────────
def clarify_question(question: str) -> str:
    """
    Returns a clarification question to surface to the user.
    The agentic loop in app.py intercepts this and pauses for user input.
    """
    return json.dumps({"clarification_needed": True, "question": question})


# ── 2. filter_registrations ──────────────────────────────────────────────────
def filter_registrations(filters: dict) -> str:
    df = _get("registrations")
    df = _apply_filters(df, filters)
    _datasets["filtered_registrations"] = df
    return _df_summary(df, "filtered_registrations")


# ── 3. filter_logins ─────────────────────────────────────────────────────────
def filter_logins(filters: dict, months_back: int | None = None) -> str:
    df = _get("logins")
    cutoff = _months_back_cutoff(months_back)
    if cutoff is not None:
        df = df[df["Timestamp"] >= cutoff]
    df = _apply_filters(df, filters)
    _datasets["filtered_logins"] = df
    return _df_summary(df, "filtered_logins")


# ── 4. join_sheets ───────────────────────────────────────────────────────────
def join_sheets(
    login_filters: dict,
    registration_filters: dict,
    months_back: int | None = None,
) -> str:
    logins = _get("logins")
    regs = _get("registrations")

    cutoff = _months_back_cutoff(months_back)
    if cutoff is not None:
        logins = logins[logins["Timestamp"] >= cutoff]

    logins = _apply_filters(logins, login_filters)
    regs = _apply_filters(regs, registration_filters)

    joined = logins.merge(regs, on="Username", how="inner", suffixes=("_login", "_reg"))
    # Keep login Timestamp accessible as 'Timestamp' for month-grouping
    if "Timestamp_login" in joined.columns:
        joined["Timestamp"] = joined["Timestamp_login"]
    _datasets["joined"] = joined
    return _df_summary(joined, "joined")


# ── 5. group_and_count ───────────────────────────────────────────────────────
def group_and_count(
    dataset: str,
    group_by: list[str],
    count_col: str | None = None,
    months_back: int | None = None,
) -> str:
    df = _get(dataset)

    cutoff = _months_back_cutoff(months_back)
    if cutoff is not None and "Timestamp" in df.columns:
        df = df[df["Timestamp"] >= cutoff]

    # If grouping by month and Timestamp exists, derive a Month column
    processed_group_by = []
    for g in group_by:
        if g.lower() == "month" and "Timestamp" in df.columns:
            df = df.copy()
            df["Month"] = df["Timestamp"].dt.to_period("M").astype(str)
            processed_group_by.append("Month")
        else:
            processed_group_by.append(g)

    missing = [g for g in processed_group_by if g not in df.columns]
    if missing:
        return json.dumps({"error": f"Columns not found: {missing}. Available: {list(df.columns)}"})

    if count_col and count_col in df.columns:
        grouped = df.groupby(processed_group_by)[count_col].count().reset_index()
        grouped.rename(columns={count_col: "count"}, inplace=True)
    else:
        grouped = df.groupby(processed_group_by).size().reset_index(name="count")

    result_key = f"grouped_{'_'.join(processed_group_by)}"
    _datasets[result_key] = grouped

    return json.dumps({
        "dataset_key": result_key,
        "row_count": len(grouped),
        "data": json.loads(grouped.to_json(orient="records", date_format="iso")),
    })


# ── 6. create_bar_chart ──────────────────────────────────────────────────────
_DARK_COLORS = [
    "#E86B3A", "#4CC9F0", "#7BED9F", "#FFC06E",
    "#A78BFA", "#F472B6", "#34D399", "#60A5FA",
]

_CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1A1D27",
    plot_bgcolor="#1A1D27",
    font_color="#FAFAFA",
    title_font_color="#FAFAFA",
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=50, l=10, r=10, b=10),
    colorway=_DARK_COLORS,
)


def create_bar_chart(
    data_json: str,
    x: str,
    y: str,
    color: str | None = None,
    title: str = "",
    barmode: str = "group",
) -> str:
    df = pd.read_json(StringIO(data_json), orient="records")
    fig = px.bar(
        df, x=x, y=y, color=color, title=title, barmode=barmode,
        template="plotly_dark", color_discrete_sequence=_DARK_COLORS,
    )
    fig.update_layout(**_CHART_LAYOUT)
    fig.update_traces(marker_line_width=0)
    return _save_chart(fig)


# ── 7. create_line_chart ─────────────────────────────────────────────────────
def create_line_chart(
    data_json: str,
    x: str,
    y: str,
    color: str | None = None,
    title: str = "",
) -> str:
    df = pd.read_json(StringIO(data_json), orient="records")
    fig = px.line(
        df, x=x, y=y, color=color, title=title, markers=True,
        template="plotly_dark", color_discrete_sequence=_DARK_COLORS,
    )
    fig.update_layout(**_CHART_LAYOUT)
    fig.update_traces(line_width=2.5, marker_size=7)
    return _save_chart(fig)


# ── 8. create_pie_chart ──────────────────────────────────────────────────────
def create_pie_chart(
    data_json: str,
    names: str,
    values: str,
    title: str = "",
) -> str:
    df = pd.read_json(StringIO(data_json), orient="records")
    fig = px.pie(
        df, names=names, values=values, title=title, hole=0.35,
        template="plotly_dark", color_discrete_sequence=_DARK_COLORS,
    )
    fig.update_layout(**_CHART_LAYOUT)
    fig.update_traces(textfont_color="#FAFAFA", pull=0.02)
    return _save_chart(fig)


def _save_chart(fig) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=tempfile.gettempdir())
    fig.write_json(tmp.name)
    tmp.close()
    return json.dumps({"chart_path": tmp.name})


# ── 9. summarise_dataframe ───────────────────────────────────────────────────
def summarise_dataframe(data_json: str, columns: list[str]) -> str:
    df = pd.read_json(StringIO(data_json), orient="records")
    result = {}
    for col in columns:
        if col not in df.columns:
            result[col] = "column not found"
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            result[col] = {
                "count": int(s.count()),
                "mean": round(float(s.mean()), 2),
                "min": float(s.min()),
                "max": float(s.max()),
                "sum": float(s.sum()),
            }
        else:
            vc = s.value_counts().head(5)
            result[col] = {
                "count": int(s.count()),
                "unique": int(s.nunique()),
                "top_values": vc.to_dict(),
            }
    return json.dumps(result)


# ── 10. get_column_values ─────────────────────────────────────────────────────
def get_column_values(sheet: str, column: str) -> str:
    df = _get(sheet)
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not in '{sheet}'. Available: {list(df.columns)}"})
    vals = df[column].dropna().unique().tolist()
    # Convert non-serialisable types
    vals = [str(v) if not isinstance(v, (str, int, float, bool)) else v for v in vals]
    return json.dumps({"sheet": sheet, "column": column, "unique_values": vals[:100]})


# ── Dispatch table used by app.py ─────────────────────────────────────────────
TOOL_FUNCTIONS: dict[str, callable] = {
    "clarify_question": clarify_question,
    "filter_registrations": filter_registrations,
    "filter_logins": filter_logins,
    "join_sheets": join_sheets,
    "group_and_count": group_and_count,
    "create_bar_chart": create_bar_chart,
    "create_line_chart": create_line_chart,
    "create_pie_chart": create_pie_chart,
    "summarise_dataframe": summarise_dataframe,
    "get_column_values": get_column_values,
}
