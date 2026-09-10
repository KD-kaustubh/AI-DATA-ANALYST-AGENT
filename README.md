# AI Data Analyst Agent

An AI-powered data analyst you can talk to: upload a CSV or XLSX dataset,
ask questions about it in plain English, and get answers backed by real
computation, not a language model's guess. A bounded agent picks from a
fixed set of pandas operations, runs the one that fits, and explains the
verified result — with follow-up questions handled in a running
conversation.

**🔗 Live Demo:** **[ai-data-analyst-ui-tfvu.onrender.com](https://ai-data-analyst-ui-tfvu.onrender.com)**

| Link | URL |
| --- | --- |
| UI (start here) | <https://ai-data-analyst-ui-tfvu.onrender.com> |
| API | <https://ai-data-analyst-api-8b3n.onrender.com> |
| API health | <https://ai-data-analyst-api-8b3n.onrender.com/api/health> |

The API and UI are deployed as two separate services on Render, in the same
containers used locally (`Dockerfile.api` / `Dockerfile.ui`). The free tier
sleeps after inactivity, so the first request after a while can take a few
seconds to wake up.

**Status:** deployed and verified end to end — dataset upload, profiling,
natural-language analysis, follow-up questions, and grounded answers have
all been exercised against the live app running on Groq
(`openai/gpt-oss-120b`). 342 automated tests, 0 known failures.

## The problem this solves

Handing a language model a spreadsheet and asking it to "just answer" a
numeric question is unreliable — models are fluent, but they are not
calculators, and they will produce a plausible-sounding wrong number as
readily as a right one. This project keeps the two jobs separate: the model
decides *what to compute* by picking from a small, explicit set of tools;
pandas does the *actual computing*. The model never touches the data
directly, and a lightweight grounding check flags any figure in its final
answer that doesn't trace back to a verified result.

## Features

- CSV/XLSX upload with an immediate structured profile (row/column counts,
  dtypes, missing values, duplicates, detected date columns)
- Natural-language questions, answered through structured LLM tool calling
  — the model requests one of seven deterministic pandas operations, never
  writes or runs code itself
- Multi-step agent reasoning with a hard cap (max 5 tool calls per
  question), so a confused model can't loop forever
- Conversation / follow-up questions ("...and which one is highest?")
  using bounded, non-growing history
- The agent asks for clarification instead of guessing when a question is
  genuinely ambiguous
- Evidence tracking and grounding validation on every answer, flagging any
  number that doesn't appear in the computed evidence
- Charts and visualizations (bar, line, histogram, scatter, box,
  correlation heatmap) available as a library
- Gemini and Groq provider support behind one interface — switching is one
  environment variable
- A safe, explicit tool registry: the model can only select from a fixed
  list, and no arbitrary Python execution is possible anywhere in the path
- FastAPI backend, Streamlit frontend, both containerized with Docker and
  deployed on Render as separate services

## Architecture

```
                     Browser
                        │
                        ▼
                 Streamlit UI  (app.py, :8501)
                        │  HTTP (api_client.py)
                        ▼
                 FastAPI backend  (:8000)
                        │
                        ▼
              Conversation / Agent loop  ──── bounded, max 5 tool calls
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
      LLM provider              Tool dispatcher
   (Gemini or Groq,                  │
    one interface)          only registered tools,
           │                 only declared arguments
           ▼                         ▼
   structured tool request ──> pandas analysis engine
                                     │
                                     ▼
                            verified AnalysisResult
                                     │
                        ┌────────────┴────────────┐
                        ▼                          ▼
                 grounding check           LLM writes final answer
                (flags unsupported          (from the verified result
                 numbers, doesn't            only — never raw data)
                 invent a "fix")
```

Every arrow from "LLM provider" back down is a **request for a named tool
with typed arguments**, never code or a query string. The dispatcher looks
the name up in a fixed registry and rejects anything else — there is no
`eval`, no `exec`, no dynamic import driven by model output, anywhere in
this path.

**In production:** Streamlit + FastAPI are each deployed to Render as a
separate Docker web service; the LLM provider is Groq
(`openai/gpt-oss-120b`). Locally, the same two containers run through
`docker compose`, and either Gemini or Groq can be selected with one
environment variable.

| Layer | Module | Role |
|---|---|---|
| Loading | `analyst.loader` | CSV/XLSX in, validated DataFrame out |
| Profiling | `analyst.profiling` | schema, gaps, duplicates, date detection |
| Analysis | `analyst.analysis` | seven deterministic pandas operations |
| Charts | `analyst.charts` | six Matplotlib figures (library only, not yet an agent tool) |
| Tools | `analyst.tools` | the registry and the safe dispatcher |
| Agent | `analyst.agent` | the bounded multi-step tool loop |
| Grounding | `analyst.grounding` | checks the final answer's numbers against evidence |
| Providers | `analyst.gemini`, `analyst.groq` | one `LLMClient` interface, two implementations |
| Session | `analyst.conversation` | in-memory follow-up context |
| API | `analyst.api` | FastAPI over all of the above |
| UI | `app.py` | Streamlit; talks to the API over HTTP only |

## Tech stack

| Concern | Choice |
|---|---|
| Language | Python 3.12 |
| Data | pandas, numpy |
| Charts | Matplotlib |
| LLM providers | Google Gemini (`google-genai` SDK), Groq (`groq` SDK) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Validation | Pydantic |
| Testing | pytest, FastAPI's `TestClient` |
| Containers | Docker, Docker Compose |
| Deployment | Render (Docker-native free web services) |

## Project structure

```
src/analyst/
├── loader.py, validation.py, profiling.py   # Phase 1 — dataset foundation
├── analysis.py, charts.py, conversion.py     # Phase 2 — analysis & viz
├── tools.py, prompts.py                      # tool registry + prompts
├── llm.py, gemini.py, groq.py                # provider abstraction
├── agent.py, conversation.py, grounding.py   # the agent loop
└── api/                                      # FastAPI app
    ├── main.py       # routes, error handling, CORS
    ├── schemas.py     # Pydantic request/response models
    └── store.py       # in-memory datasets & sessions

app.py            # Streamlit UI
api_client.py     # the UI's only path to the backend (plain HTTP)
tests/            # 342 tests, one file per module above
Dockerfile.api, Dockerfile.ui, docker-compose.yml, render.yaml
```

## How the agent works

1. The UI (or a direct API call) sends a question plus a dataset id.
2. The agent builds a compact **schema summary** from the dataset's profile
   — column names, types, missing counts — never the raw rows.
3. It asks the LLM for one JSON action: `call_tool` (with a tool name and
   arguments), `answer` (evidence already suffices, or the dataset genuinely
   can't answer this), or `clarification` (the question is ambiguous).
4. A `call_tool` request goes through the dispatcher, which checks the tool
   is registered and its arguments match the declared types before running
   the real pandas function. A bad request — unknown tool, wrong argument,
   a hallucinated column — is fed back to the model as an error, not raised;
   it gets one more step to correct itself.
5. Steps 3–4 repeat, up to 5 times, with every prior step (including
   failures) shown back to the model.
6. Once the model has enough evidence, a **separate** prompt asks it to
   write the final answer using only the verified results — the schema and
   evidence, never the dataset itself.
7. The answer's numbers are checked against the evidence (`grounding.py`).
   A mismatch is reported (`grounding.is_grounded: false` and which figures
   are unsupported), not silently hidden or "corrected".

## Supported analysis tools

`filter_rows`, `sort_rows`, `group_aggregate`, `describe_numeric`,
`value_counts`, `correlation`, `group_by_period` — see `analyst.tools` for
the exact arguments each one accepts. All seven are deterministic pandas
operations; the model can only select from this fixed list.

## LLM provider configuration

Both Gemini and Groq are supported behind one `LLMClient` interface.
`create_client()` picks a provider: `LLM_PROVIDER` if set, otherwise
whichever API key is present (Gemini wins if both are).

- **Currently deployed:** Groq, model `openai/gpt-oss-120b` — this is what
  `/api/health` on the live deployment reports, and what the live demo runs
  on.
- **Gemini** (`gemini-2.5-flash` by default) is fully implemented and
  unit-tested against a stubbed SDK; it works locally with a
  `GOOGLE_API_KEY` but is not the provider currently deployed.

Switching provider is one environment variable (`LLM_PROVIDER=gemini` or
`groq`), no code change. Adding a third provider means one class
implementing `generate()` plus one entry in `PROVIDER_SETTINGS`.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; source .venv/bin/activate on Unix
pip install -r requirements.txt
cp .env.example .env            # then add your API key
```

`pip install -r requirements.txt` installs the project itself in editable mode
(`-e .[dev]`), which is what puts `analyst` on the import path. Without it,
`uvicorn analyst.api:app` fails with `No module named 'analyst'`.

Run the backend and the UI in two terminals, **each with the virtualenv
activated** — otherwise `uvicorn` and `streamlit` resolve to a different Python
that has neither the project nor its dependencies:

```bash
uvicorn analyst.api:app --reload     # http://127.0.0.1:8000
streamlit run app.py                 # http://localhost:8501
```

If a command still misbehaves, run it through the venv explicitly:

```bash
.venv\Scripts\python -m uvicorn analyst.api:app --reload
.venv\Scripts\python -m streamlit run app.py
```

Interactive API docs are at `http://127.0.0.1:8000/docs`.

## Run with Docker

The same two services run in containers, with no change to their behavior —
same in-memory state, same `/api/*` routes, same UI.

```bash
cp .env.example .env      # then add your API key; compose reads this file
docker compose up --build # build both images and start the stack
```

- Streamlit: `http://localhost:8501`
- API: `http://localhost:8000` (published for local debugging; the UI
  container reaches it internally as `http://api:8000`, the compose service
  name — never `127.0.0.1` or `localhost`, which would mean "this container")
- API health: `curl http://localhost:8000/api/health`

Stop everything with:

```bash
docker compose down
```

If port 8000 or 8501 is already taken on your machine, override the host
side only — the containers still talk to each other on their normal ports:

```bash
API_HOST_PORT=8001 docker compose up --build
```

**Secrets never enter an image.** `Dockerfile.api` and `Dockerfile.ui` never
`COPY` `.env` — `.dockerignore` excludes it, and neither Dockerfile
references it. Compose loads it only at *container start*, via
`env_file: [{path: .env, required: false}]` on the `api` service, so a
clean checkout with no `.env` still builds and starts (just unconfigured).
The `ui` service never receives a key at all — it only ever calls the API
over HTTP, the same as it does outside Docker.

Both images run as a non-root user, and the API container has a
Docker-level `healthcheck` against `/api/health` that `ui` waits on before
starting. As outside Docker, everything is in-memory: restarting a
container clears its datasets and sessions. There is no database, cache, or
persistent volume.

## Deployment

**Live now** on [Render](https://render.com), deployed as two separate
Docker web services from the same `Dockerfile.api` / `Dockerfile.ui` used
locally — no code written specifically for deployment, beyond honoring a
platform-provided `$PORT`:

- UI: https://ai-data-analyst-ui-tfvu.onrender.com
- API: https://ai-data-analyst-api-8b3n.onrender.com

The API and UI are independent services with no shared process or disk —
the UI reaches the API over plain HTTPS via `ANALYST_API_URL`, the same way
it reaches `http://api:8000` inside Docker Compose.

To redeploy this repository to your own Render account, using the included
`render.yaml` Blueprint:

1. Fork or push this repository to your own GitHub.
2. On Render: **New +** → **Blueprint**, select the repository. Render
   reads `render.yaml` and creates both services.
3. Open the api service → **Environment**, and set `LLM_PROVIDER` plus one
   provider key (`GROQ_API_KEY` or `GOOGLE_API_KEY`). These are entered in
   Render's dashboard, never committed — `render.yaml` marks them
   `sync: false` for exactly this reason.
4. Once the api service has deployed, copy its URL.
5. Open the ui service → **Environment**, set `ANALYST_API_URL` to that
   URL, and redeploy.

This is intentionally the simplest deployment that fits: no Kubernetes, no
Terraform, no CI/CD pipeline, no reverse proxy, no database. Two
containers, one env var connecting them, matching the architecture used
everywhere else in this project. The trade-off of Render's free tier: both
services idle-sleep after inactivity and take a few seconds to wake on the
next request.

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
| `PORT` | no | overrides the container's listen port (set by most deploy platforms) |

Keys are read from the environment or `.env`. `.env` is git-ignored, and no key
is ever logged, returned by the API, or included in an error message.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness, plus the active provider/model |
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

## Example questions

Verified against a small employee dataset (`id, name, age, department,
salary, country`) on the **live deployment**:

- "What is the average salary?" → one `describe_numeric` call
- "What is the average salary by department?" → one `group_aggregate` call
- "Which department has the highest average salary?" → answered as a
  follow-up, reusing the previous answer's evidence rather than
  re-running an identical query
- "How many employees are in each department?" → one `value_counts` call

A few more that the same tools support, not re-run in this pass:

- "Show the distribution of the department column." → `value_counts`
- "What is the correlation between salary and age?" → `correlation`
- "How has revenue changed over time?" (on a dataset with a date column)
  → `group_by_period`

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

```python
from analyst import answer_question, create_client, load_dataset

frame = load_dataset("data/sales.csv")
answer = answer_question(frame, "Which region earns the most?", create_client())

print(answer.text)       # the reply in plain language
print(answer.kind)       # answer, clarification or incomplete
print(answer.steps)      # every tool call, with its verified result
print(answer.grounding)  # which figures were traced back to the evidence
```

### Conversations

```python
from analyst import Conversation, create_client, load_dataset

session = Conversation(load_dataset("data/sales.csv"), create_client())
session.ask("What is the average revenue by category?")
session.ask("Which one is highest?")     # read against the previous turn
```

## Testing

```bash
pytest          # the whole suite; warnings are treated as errors
```

**342 tests, all passing**, covering every module: dataset loading and
validation, profiling, the seven analysis operations, all six charts, both
providers (against stubbed SDKs — no network or API key needed), the tool
registry and dispatcher, the agent loop, conversation/follow-up behavior,
grounding, and the full HTTP API (using FastAPI's `TestClient` with a fake
LLM client, so the suite never makes a live provider call). Edge cases with
dedicated tests include: empty datasets, invalid/unsupported files,
malformed LLM JSON, unknown tools, invalid or hallucinated arguments,
numeric/non-numeric mismatches, provider failures including rate limiting,
empty LLM responses, the agent's step limit, clarification requests,
follow-up questions, and conversation reset.

## Security considerations

- **No arbitrary code execution.** The model can only request a tool from
  an explicit registry, with arguments checked against a declared type and
  shape before anything runs. There is no `eval`, `exec`, dynamic import,
  or shell call anywhere in the request-handling path.
- **Uploads are validated, not trusted.** Extension and size are checked
  before parsing; the file is written to a generated temporary path
  (the client's filename is never used as a path) and deleted immediately
  after parsing, whether parsing succeeded or not.
- **Errors are sanitized.** Provider failures, tool errors, and unexpected
  exceptions are all mapped to a fixed, generic message before reaching the
  client — a raw traceback or provider error body is never returned. A 429
  from the provider is reported as a friendly rate-limit message, not
  forwarded verbatim.
- **Secrets stay out of the build.** Neither Dockerfile ever `COPY`s
  `.env`; both are excluded via `.dockerignore`. Compose supplies them at
  container *start*, not build, time. Both containers run as a non-root
  user.
- **Known, honest limitation:** there is no authentication. Anyone who can
  reach a deployed instance can upload a dataset and ask questions against
  it, and there's no per-user isolation between sessions beyond a random
  session id. This is appropriate for a portfolio/demo deployment, not for
  hosting anyone else's real data.

## Design decisions

- **Structured tool calls, never generated code.** Letting a model write
  and run arbitrary pandas/Python would be far more flexible, and far less
  safe. A fixed tool registry with typed arguments is a small, auditable
  surface — the trade-off is deliberate.
- **Grounding is a check, not a guarantee.** `grounding.py` flags numbers in
  the final answer that don't appear anywhere in the verified evidence. It
  catches invented or miscalculated figures; it cannot judge whether the
  model's wording is a fair reading of the data, and it doesn't block an
  ungrounded answer — it reports it.
- **The agent loop is bounded, not agentic-without-limit.** A hard cap (5
  tool calls) means a confused model degrades to a clear "couldn't finish"
  result instead of looping or running up cost.
- **In-memory, not a database.** Datasets and sessions live in process
  memory. For a single-instance portfolio deployment this is simpler and
  honest about its own limits, rather than adding persistence the project
  doesn't otherwise need.

## Current limitations

- Datasets and sessions are in-memory only; restarting the API loses both.
- No authentication, rate limiting or multi-user isolation. Local/demo use.
- Uploads are held in memory while parsed, hence the 10 MB cap.
- The agent runs at most 5 tools per question, then reports `incomplete`.
- Charts exist as a library (`analyst.charts`) but are not agent tools; the UI
  draws a bar chart only when a result is one label column plus one measure.
- Grounding flags figures missing from the evidence but does not block them.
- Conversation history keeps the last 3 turns; older turns lose their rows.
- Docker images reuse one dependency list for both services (see
  `Dockerfile.ui`'s comment), so the UI image carries a few packages
  (FastAPI, matplotlib, the provider SDKs) it never imports at runtime.
- Deployed on a free tier, the API and UI services will idle-sleep and take
  a few seconds to wake on the first request after inactivity.
- Gemini's live API was not re-exercised in this pass (Groq was); Gemini
  remains covered only by its stubbed-SDK unit tests.

## License

MIT — see [LICENSE](LICENSE).
