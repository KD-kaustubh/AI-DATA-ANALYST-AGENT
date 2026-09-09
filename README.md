# AI Data Analyst Agent

An AI-powered data analysis assistant that allows users to interact with structured datasets using natural language.

## Status

🚧 In Development

## Planned Capabilities

- Dataset ingestion
- Automated data profiling
- Natural-language data analysis
- Pandas-based analytical tools
- AI-generated insights
- Data visualizations
- Conversational follow-up questions
- API and web interface

## Usage

Load a CSV or XLSX file and inspect it. Profiling only reads the data — it
never cleans, converts or drops anything.

```python
from analyst import load_dataset, profile_dataset

frame = load_dataset("data/sales.csv")
profile = profile_dataset(frame)

print(profile.row_count, profile.duplicate_row_count)
print(profile.warnings)
print(profile.to_dict())  # plain dicts and lists, ready to serialise
```

Loading failures raise a subclass of `DatasetError`: `DatasetNotFoundError`,
`UnsupportedFileTypeError`, `EmptyDatasetError` or `DatasetReadError`.

### Analysis

Analysis operations take a DataFrame and return an `AnalysisResult` holding the
operation, the columns involved, the parameters used, the resulting rows and
some metadata. Results are JSON-serialisable and never modify the input frame.

```python
from analyst import Condition, filter_rows, group_aggregate, correlation

filter_rows(frame, [Condition("region", "==", "North"), Condition("units", ">=", 8)])
group_aggregate(frame, "region", {"revenue": ["sum", "mean"]})
correlation(frame, ["units", "revenue"])
```

Available operations: `filter_rows`, `sort_rows`, `group_aggregate`,
`describe_numeric`, `value_counts`, `correlation` and `group_by_period`.
Filters are built from structured `Condition` values, never from expressions,
so no caller-supplied code is ever evaluated.

### Charts

```python
from analyst import bar_chart, figure_to_png_bytes

figure = bar_chart(result, "region", "revenue_sum")
png = figure_to_png_bytes(figure)
```

Chart builders return a Matplotlib `Figure` and write nothing to disk. Available:
`bar_chart`, `line_chart`, `histogram`, `scatter_plot`, `box_plot` and
`correlation_heatmap`.

### Asking questions

Copy `.env.example` to `.env` and add a key for either provider, then let a
model pick the tool while pandas does the arithmetic.

```python
from analyst import answer_question, create_client, load_dataset

frame = load_dataset("data/sales.csv")
answer = answer_question(frame, "Which region earns the most?", create_client())

print(answer.text)       # the reply in plain language
print(answer.kind)       # answer, clarification or incomplete
print(answer.steps)      # every tool call, with its verified result
print(answer.grounding)  # which figures were traced back to the evidence
```

The agent works one step at a time: it picks a tool, sees the verified result,
and may call another before answering. It runs at most `max_steps` tools
(default 5) and then stops with `kind="incomplete"`, so a question can never
loop. A failed step is handed back so the model can fix its arguments; a
provider outage is raised.

### Conversations

```python
from analyst import Conversation, create_client, load_dataset

session = Conversation(load_dataset("data/sales.csv"), create_client())
session.ask("What is the average revenue by category?")
session.ask("Which one is highest?")     # read against the previous turn
```

State is in memory only. The model sees the last few turns, and only the
newest keeps its result rows, so the prompt cannot grow with the conversation.
When a follow-up is ambiguous the agent returns `kind="clarification"` with a
question rather than guessing.

Gemini and Groq are interchangeable. `create_client()` reads `LLM_PROVIDER`
when set, otherwise it uses whichever key is configured, preferring Gemini if
both are. Pass a name to be explicit: `create_client("groq")`. Adding another
provider means one class plus one entry in `PROVIDER_SETTINGS`.

The model never computes values. It chooses one registered tool and its
arguments; the dispatcher rejects anything else, runs the real Phase 2
function, and hands the verified result back for wording. There is no code
execution, and unregistered tools cannot run.

## Development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```
