# AI Data Analyst Agent

An AI-powered data analysis assistant that allows users to interact with structured datasets using natural language.

## Status

🚧 In development. The library, HTTP API and web UI all work locally.

## Architecture

```
Streamlit UI  ──HTTP──>  FastAPI  ──>  Agent  ──>  LLM (Gemini / Groq)
                                        │
                                        v
                                 Tool dispatcher
                                        │
                                        v
                                 pandas analysis
                                        │
                                        v
                                verified results
```

The model chooses which analysis to run; **pandas computes every number**. The
dispatcher only runs registered functions with declared arguments, so there is
no code execution anywhere in the loop.

| Layer | Module | Role |
|---|---|---|
| Loading | `analyst.loader` | CSV/XLSX in, validated DataFrame out |
| Profiling | `analyst.profiling` | schema, gaps, duplicates, date detection |
| Analysis | `analyst.analysis` | seven deterministic pandas operations |
| Charts | `analyst.charts` | six Matplotlib figures |
| Agent | `analyst.agent` | bounded multi-step tool loop |
| Session | `analyst.conversation` | in-memory follow-up context |
| API | `analyst.api` | FastAPI over all of the above |
| UI | `app.py` | Streamlit, talks HTTP only |

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use source .venv/bin/activate on Unix
pip install -r requirements.txt
cp .env.example .env            # then add your API key
```

Run the backend and the UI in two terminals:

```bash
uvicorn analyst.api:app --reload     # http://127.0.0.1:8000
streamlit run app.py                 # http://localhost:8501
```

Interactive API docs are at `http://127.0.0.1:8000/docs`.

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | for Gemini | Gemini credentials |
| `GROQ_API_KEY` | for Groq | Groq credentials |
| `LLM_PROVIDER` | no | `gemini` or `groq`; otherwise whichever key is set |
| `MODEL_NAME` | no | Gemini model, default `gemini-2.5-flash` |
| `GROQ_MODEL_NAME` | no | Groq model, default `llama-3.3-70b-versatile` |
| `CORS_ORIGINS` | no | comma-separated origins allowed to call the API |
| `ANALYST_API_URL` | no | where the UI looks for the backend |

Keys are read from the environment or `.env`. `.env` is git-ignored, and no key
is ever logged, returned by the API, or included in an error message.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness |
| POST | `/api/datasets` | upload a CSV/XLSX, get its profile |
| GET | `/api/datasets/{id}` | profile of an uploaded dataset |
| POST | `/api/sessions` | open a conversation over a dataset |
| GET | `/api/sessions/{id}` | session state |
| POST | `/api/sessions/{id}/questions` | ask, with follow-up context |
| DELETE | `/api/sessions/{id}` | end a session |
| POST | `/api/analyze` | one-off question, no conversation |

```bash
curl -F "file=@data/sales.csv" http://127.0.0.1:8000/api/datasets

curl -X POST http://127.0.0.1:8000/api/sessions \
     -H "Content-Type: application/json" -d '{"dataset_id":"<id>"}'

curl -X POST http://127.0.0.1:8000/api/sessions/<sid>/questions \
     -H "Content-Type: application/json" \
     -d '{"question":"Which region earns the most?"}'
```

Supported uploads are **CSV and XLSX**, up to **10 MB**. Responses carry the
dataset profile and the verified result rows, never the whole DataFrame.

**Datasets and sessions live in the API process memory and are lost when the
server restarts.** There is no database and nothing is written to disk.

## Library usage

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
arguments; the dispatcher rejects anything else, runs the real analysis
function, and hands the verified result back for wording. There is no code
execution, and unregistered tools cannot run.

## Development

```bash
pytest          # the whole suite; warnings are errors
```

## Limitations

- Datasets and sessions are in-memory only; restarting the API loses both.
- No authentication, rate limiting or multi-user isolation. Local use only.
- Uploads are held in memory while parsed, hence the 10 MB cap.
- The agent runs at most 5 tools per question, then reports `incomplete`.
- Charts exist as a library (`analyst.charts`) but are not agent tools; the UI
  draws a bar chart only when a result is one label column plus one measure.
- Grounding flags figures missing from the evidence but does not block them.
- Conversation history keeps the last 3 turns; older turns lose their rows.
