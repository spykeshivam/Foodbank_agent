"""
Tool schemas for the Foodbank AI agent — google-genai SDK format.

Each entry is a dict compatible with types.FunctionDeclaration(
    name=..., description=..., parameters_json_schema=...
).
The `parameters_json_schema` field accepts plain OpenAPI / JSON Schema dicts,
so we keep standard JSON Schema types (lowercase strings, nullable via "null").
"""

# Raw JSON-Schema dicts for each tool.
# app.py converts these into google.genai types.Tool objects at startup.
TOOL_SCHEMAS = [
    {
        "name": "clarify_question",
        "description": (
            "Ask the admin a clarifying question when their query is ambiguous "
            "(e.g. 'past few months' with no number specified). "
            "IMPORTANT: calling this tool pauses the agentic loop — the question "
            "is shown to the user and you will receive their answer before continuing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarifying question to ask the admin.",
                }
            },
            "required": ["question"],
        },
    },
    {
        "name": "filter_registrations",
        "description": (
            "Filter the registrations DataFrame by any column/value combination. "
            "Matching is case-insensitive substring for string columns. "
            "Returns a JSON summary with row count and first 5 rows. "
            "Filtered result stored as 'filtered_registrations'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "object",
                    "description": (
                        "Dict of {column_name: value} to filter by. "
                        "Example: {\"Sex\": \"Female\", \"Dietary Requirements\": \"Halal\"}"
                    ),
                }
            },
            "required": ["filters"],
        },
    },
    {
        "name": "filter_logins",
        "description": (
            "Filter the logins DataFrame. If months_back is provided, restricts to "
            "the last N months from today. Returns a JSON summary. "
            "Filtered result stored as 'filtered_logins'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "object",
                    "description": "Dict of {column_name: value} to filter by.",
                },
                "months_back": {
                    "type": "integer",
                    "description": "Restrict to last N months. Omit for no time restriction.",
                },
            },
            "required": ["filters"],
        },
    },
    {
        "name": "join_sheets",
        "description": (
            "Inner-join logins to registrations on the Username column. "
            "Applies separate filter dicts to each sheet before joining. "
            "Optionally restricts logins to the last N months. "
            "Result stored as 'joined'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "login_filters": {
                    "type": "object",
                    "description": "Filters to apply to the logins sheet.",
                },
                "registration_filters": {
                    "type": "object",
                    "description": "Filters to apply to the registrations sheet.",
                },
                "months_back": {
                    "type": "integer",
                    "description": "Restrict logins to last N months. Omit for no restriction.",
                },
            },
            "required": ["login_filters", "registration_filters"],
        },
    },
    {
        "name": "group_and_count",
        "description": (
            "Group a dataset by one or more columns and return counts. "
            "Use 'month' as a group_by value to group by calendar month (requires Timestamp). "
            "dataset can be: 'registrations', 'logins', 'joined', 'filtered_registrations', "
            "'filtered_logins', or any previously stored key. "
            "Returns JSON with grouped data and a dataset_key for chart tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset": {
                    "type": "string",
                    "description": "Name of the dataset to group.",
                },
                "group_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to group by. Use 'month' for monthly aggregation.",
                },
                "count_col": {
                    "type": "string",
                    "description": "Column to count (optional). If omitted, counts rows.",
                },
                "months_back": {
                    "type": "integer",
                    "description": "Further restrict to last N months before grouping.",
                },
            },
            "required": ["dataset", "group_by"],
        },
    },
    {
        "name": "create_bar_chart",
        "description": (
            "Create a Plotly bar chart from a JSON records string. "
            "Returns a file path — the Streamlit app renders it automatically. "
            "Pass the 'data' array from group_and_count serialised as JSON."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "data_json": {
                    "type": "string",
                    "description": "JSON array of records, e.g. '[{\"Month\":\"2025-01\",\"count\":12}]'",
                },
                "x": {"type": "string", "description": "Column name for the x-axis."},
                "y": {"type": "string", "description": "Column name for the y-axis."},
                "color": {"type": "string", "description": "Column for colour grouping (optional)."},
                "title": {"type": "string", "description": "Chart title."},
                "barmode": {
                    "type": "string",
                    "description": "Bar mode: 'group', 'stack', or 'relative'. Default 'group'.",
                },
            },
            "required": ["data_json", "x", "y", "title", "barmode"],
        },
    },
    {
        "name": "create_line_chart",
        "description": "Create a Plotly line/trend chart from a JSON records string. Returns a file path.",
        "parameters": {
            "type": "object",
            "properties": {
                "data_json": {"type": "string", "description": "JSON array of records."},
                "x": {"type": "string", "description": "Column name for the x-axis."},
                "y": {"type": "string", "description": "Column name for the y-axis."},
                "color": {"type": "string", "description": "Column for colour grouping (optional)."},
                "title": {"type": "string", "description": "Chart title."},
            },
            "required": ["data_json", "x", "y", "title"],
        },
    },
    {
        "name": "create_pie_chart",
        "description": "Create a Plotly pie/donut chart from a JSON records string. Returns a file path.",
        "parameters": {
            "type": "object",
            "properties": {
                "data_json": {"type": "string", "description": "JSON array of records."},
                "names": {"type": "string", "description": "Column for slice labels."},
                "values": {"type": "string", "description": "Column for slice sizes."},
                "title": {"type": "string", "description": "Chart title."},
            },
            "required": ["data_json", "names", "values", "title"],
        },
    },
    {
        "name": "summarise_dataframe",
        "description": (
            "Return descriptive statistics for specific columns in a JSON dataset. "
            "Numeric columns: count/mean/min/max/sum. "
            "Categorical columns: count/unique/top_values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "data_json": {"type": "string", "description": "JSON array of records."},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of column names to summarise.",
                },
            },
            "required": ["data_json", "columns"],
        },
    },
    {
        "name": "get_column_values",
        "description": (
            "Return the unique values present in a column so you know what filter "
            "values are valid before calling filter_registrations or filter_logins."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sheet": {
                    "type": "string",
                    "description": "Which sheet to inspect: 'registrations' or 'logins'.",
                },
                "column": {"type": "string", "description": "Column name to inspect."},
            },
            "required": ["sheet", "column"],
        },
    },
]
