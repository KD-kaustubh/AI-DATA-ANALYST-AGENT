# AI Data Analyst Agent — Technical Project Handoff

**Status:** Complete and deployed. All application phases finished; the
system is live and has been exercised end to end against the real Groq API.

**Last verified state:** 2026-09-10 — full test suite green (342/342), live
health check returned the expected payload, and a real multi-turn
conversation (initial question + follow-up) was verified against the
deployed API.

**Repository:** <https://github.com/KD-kaustubh/AI-DATA-ANALYST-AGENT.git>

**Live UI:** <https://ai-data-analyst-ui-tfvu.onrender.com>

**Live API:** <https://ai-data-analyst-api-8b3n.onrender.com>

**Health check:** <https://ai-data-analyst-api-8b3n.onrender.com/api/health>
— verified response:

```json
{"status": "ok", "version": "0.1.0", "provider": "groq", "model": "openai/gpt-oss-120b"}
```

**Production provider / model:** Groq, `openai/gpt-oss-120b`

**Purpose of this document:** This is not the README. The README is for
visitors, recruiters and interviewers skimming the repository. This
document is for whoever (most likely the original author) comes back to
this codebase later — possibly months later — and needs to actually work
on it: extend it, debug it, redeploy it, change the provider, or explain
a design decision in an interview without re-reading every source file.
Everything below is checked against the actual source, tests, Docker
configuration, and Render configuration in this repository as of the
commit noted in [Section 35](#section-35--final-project-snapshot). Anything
that could not be confirmed this way is explicitly labeled **"Not verified
from repository."**

---

## Table of Contents

1. [Executive Summary](#section-1--executive-summary)
2. [Project History / Development Roadmap](#section-2--project-history--development-roadmap)
3. [Current System Architecture](#section-3--current-system-architecture)
4. [Repository Structure](#section-4--repository-structure)
5. [Data Ingestion and Profiling](#section-5--data-ingestion-and-profiling)
6. [Analysis Engine](#section-6--analysis-engine)
7. [Visualization Engine](#section-7--visualization-engine)
8. [LLM Architecture](#section-8--llm-architecture)
9. [Structured Tool Calling](#section-9--structured-tool-calling)
10. [Agent Loop](#section-10--agent-loop)
11. [Conversation State](#section-11--conversation-state)
12. [Evidence and Grounding](#section-12--evidence-and-grounding)
13. [FastAPI Backend](#section-13--fastapi-backend)
14. [Streamlit UI](#section-14--streamlit-ui)
15. [API Client](#section-15--api-client)
16. [Docker Architecture](#section-16--docker-architecture)
17. [Render Deployment](#section-17--render-deployment)
18. [Environment Configuration](#section-18--environment-configuration)
19. [Testing Architecture](#section-19--testing-architecture)
20. [Security Model](#section-20--security-model)
21. [Error Handling](#section-21--error-handling)
22. [Logging and Observability](#section-22--logging-and-observability)
23. [Known Limitations](#section-23--known-limitations)
24. [Common Troubleshooting](#section-24--common-troubleshooting)
25. [How to Add a New Analysis Tool](#section-25--how-to-add-a-new-analysis-tool)
26. [How to Add a New LLM Provider](#section-26--how-to-add-a-new-llm-provider)
27. [How to Modify the Agent](#section-27--how-to-modify-the-agent)
28. [How to Redeploy From Scratch](#section-28--how-to-redeploy-from-scratch)
29. [Local Development Workflow](#section-29--local-development-workflow)
30. [Git / Development Workflow](#section-30--git--development-workflow)
31. [Verified Production Checklist](#section-31--verified-production-checklist)
32. [Important File Reference](#section-32--important-file-reference)
33. [Future Improvement Ideas](#section-33--future-improvement-ideas-not-implemented)
34. [Interview / Technical Discussion Notes](#section-34--interview--technical-discussion-notes)
35. [Final Project Snapshot](#section-35--final-project-snapshot)

---

## Section 1 — Executive Summary

**What it does.** A user uploads a CSV or XLSX dataset and asks questions
about it in plain English — "What is the average salary by department?",
"Which department has the highest average salary?". The system answers with
numbers computed by pandas, not numbers guessed by a language model, and
follow-up questions work in a running conversation.

**Problem it solves.** A language model asked to "just answer" a numeric
question from raw data will produce a fluent, plausible-looking answer
whether or not the number is correct — it is not a calculator. This project
separates the two responsibilities: the LLM decides *which* deterministic
pandas operation answers the question and with what arguments; pandas
performs the actual computation; a lightweight grounding check then verifies
that every number in the model's final wording actually appears in that
computed result.

**Intended user.** Someone with a dataset (a spreadsheet export, a small
CSV) who wants to ask it questions without writing pandas code themselves —
positioned as a portfolio/demo project, not a multi-tenant SaaS product (see
[Section 20](#section-20--security-model) and
[Section 23](#section-23--known-limitations)).

**How the user interacts with it.** Through the Streamlit web UI
(`app.py`): upload a file, see its profile, type a question in a chat box,
read the answer plus (optionally) the underlying result table, a chart, and
which tool(s) were used. The UI is also a documented, testable REST API
(`analyst.api`) that can be used directly (see
[Section 13](#section-13--fastapi-backend)).

**What makes it technically interesting.**
- The LLM never touches the DataFrame. It can only request one of seven
  named, typed pandas operations from an explicit registry — there is no
  `eval`, `exec`, or dynamic code path anywhere between a model's reply and
  the data (see [Section 9](#section-9--structured-tool-calling) and
  [Section 20](#section-20--security-model)).
- The agent runs a **bounded** multi-step loop (`DEFAULT_MAX_STEPS = 5`,
  `src/analyst/agent.py`), so it can chain a few tool calls together for a
  multi-part question, correct a bad tool argument by seeing the error, and
  is mathematically guaranteed to terminate rather than loop forever.
- A **grounding check** (`src/analyst/grounding.py`) parses the numbers out
  of the model's final answer text and flags any that don't trace back to a
  computed result — it doesn't block a bad answer, but it makes an
  ungrounded claim visible rather than silent.
- Two LLM providers (Gemini, Groq) sit behind one `LLMClient` interface,
  switched with a single environment variable.

**Major technologies.** Python 3.12, pandas, NumPy, Matplotlib, FastAPI,
Pydantic, Streamlit, the `google-genai` and `groq` SDKs, Docker, Render.

**Current deployment state.** Live on Render as two independent Docker web
services (`ai-data-analyst-api`, `ai-data-analyst-ui`), configured through
`render.yaml`. Production provider is Groq, model `openai/gpt-oss-120b`.

---

## Section 2 — Project History / Development Roadmap

This is reconstructed from the actual git history (`git log --oneline
--reverse`, 14 commits, all on `main`, no branches, no merges, no force
pushes, no rewritten history) plus knowledge of what each commit's diff
actually contains. Commit hashes are real and can be checked directly with
`git show <hash>`.

| Commit | Message | Stage |
|---|---|---|
| `14e1701` | Initialize AI data analyst agent | Phase 0 — project scaffold |
| `dcf11a3` | Add dataset loading and profiling | Phase 1 — Dataset Foundation |
| `aad96fe` | Add analysis and chart tools | Phase 2 — Analysis & Visualization |
| `0718f1b` | Add LLM tool calling | Phase 3 — LLM Integration |
| `59b00ee` | Add Groq provider support | Phase 3.1 — Provider abstraction |
| `109ed04` | Add agent loop and conversation state | Phase 4 — Agent + Grounding + Conversation |
| `8320edb` | Add FastAPI backend and Streamlit UI | Phase 5 — API + UI |
| `ff84055` | Install the package properly for local startup | Phase 5 fix — packaging |
| `8c40327` | Improve rate limit error handling | Phase 5 fix — 429 handling |
| `2d6c562` | Show active AI model in UI | Phase 5 fix — provider caption |
| `4bf96ae` | Add Docker setup | Phase 6.1 — Dockerization |
| `af529ff` | Improve production configuration | Phase 6 — production readiness (logging, `$PORT`) |
| `20a99b1` | Complete deployment setup and finish documentation | Phase 6 — `render.yaml`, README, LICENSE |
| `db6886c` | Update README for deployed app | Final documentation pass |

### Phase 0 — `14e1701`

**Objective:** a clean, installable Python project skeleton.
**Key files:** `pyproject.toml`, `src/analyst/__init__.py`, `.gitignore`.
**Resulting capability:** an importable, empty `analyst` package with a
build system, ready for real modules.

### Phase 1 — Dataset Foundation (`dcf11a3`)

**Objective:** reliably get a CSV or XLSX file into a validated
`pandas.DataFrame`, plus a structured description of it.
**Key files:** `loader.py`, `validation.py`, `profiling.py`,
`conversion.py`, `errors.py` (the `DatasetError` family).
**Design decisions:** validation happens both before parsing (extension,
existence, non-zero size) and after (non-empty frame) so a header-only file
is rejected the same way an unreadable one is; datetime detection is
deliberately conservative (see [Section 5](#section-5--data-ingestion-and-profiling));
numpy/pandas values are never returned raw — `conversion.py` exists purely
to make everything JSON-safe.
**Verification:** unit tests in `test_loader.py` (12 tests) and
`test_profiling.py` (17 tests).
**Resulting capability:** `load_dataset()` → `DataFrame`; `profile_dataset()`
→ `DatasetProfile`.

### Phase 2 — Analysis & Visualization (`aad96fe`)

**Objective:** a deterministic pandas operations layer the rest of the
system can build on, plus chart builders.
**Key files:** `analysis.py` (seven operations, `Condition`,
`AnalysisResult`), `charts.py` (six chart builders).
**Design decisions:** every operation validates its own arguments and
raises a specific `AnalysisError` subclass; filters use structured
`Condition` objects, never expression strings, so there is no evaluation
surface even at this layer (this predates and is independent of the LLM
work that comes later); charts use Matplotlib's object-oriented `Figure`
API, not `pyplot`, so nothing is stored in Matplotlib's global figure
registry — safe for a long-running server process.
**Verification:** `test_analysis.py` (58 tests), `test_charts.py`
(27 tests).
**Resulting capability:** `filter_rows`, `sort_rows`, `group_aggregate`,
`describe_numeric`, `value_counts`, `correlation`, `group_by_period`, and
six chart builders — all usable as a plain Python library, no LLM involved.

### Phase 3 — LLM Integration (`0718f1b`)

**Objective:** let a language model select and call one of the Phase 2
operations, safely.
**Key files:** `tools.py` (registry, `ToolRequest`, dispatcher),
`context.py` (schema summary for the prompt), `prompts.py` (first version),
`questions.py` (this module was later renamed/superseded — see Phase 4).
**Design decision:** the model receives a `DatasetContext` built from the
`DatasetProfile` — column names, types, missing counts — never the
DataFrame itself. This is a structural guarantee: the raw rows are simply
never serialized into a prompt.
**Resulting capability:** a single natural-language question could be
answered with exactly one tool call.

### Phase 3.1 — Provider Abstraction (`59b00ee`)

**Objective:** support Groq alongside Gemini behind one interface, so the
provider is a configuration choice, not a code fork.
**Key files:** `llm.py` (`LLMClient` protocol, `LLMConfig`,
`ProviderSettings`, `resolve_provider`, `create_client`), `gemini.py`,
`groq.py`.
**Design decision:** `create_client()` imports the chosen provider's SDK
module lazily, inside the branch that needs it — so a Gemini-only install
doesn't require the `groq` package to be importable, and vice versa.
**Resulting capability:** `LLM_PROVIDER=groq` or `gemini` picks the
provider; whichever API key is present is used if `LLM_PROVIDER` is unset.

### Phase 4 — Agent + Grounding + Conversation (`109ed04`)

**Objective:** move from one tool call per question to a bounded multi-step
agent, add a grounding check on the final answer, and add multi-turn
conversation state. This is the most significant single phase — see
[Section 10](#section-10--agent-loop),
[Section 11](#section-11--conversation-state) and
[Section 12](#section-12--evidence-and-grounding) for full detail.
**Key files:** `agent.py` (replaced `questions.py`), `conversation.py`
(new), `grounding.py` (new), `tools.py` (added the `clarification` action).
**Design decisions:** the loop is capped at `DEFAULT_MAX_STEPS = 5`; a
failed tool call is fed back to the model as feedback rather than raised as
an exception, so the model gets a chance to self-correct within the step
budget; grounding is advisory (it reports, it does not veto).
**Verification:** `test_agent.py` (30 tests), `test_conversation.py`
(21 tests).
**Resulting capability:** multi-step reasoning, follow-up questions,
clarification requests, and a grounding report on every answer.

### Phase 5 — FastAPI + Streamlit (`8320edb`, plus three fix commits)

**Objective:** expose the library over HTTP and build a usable UI.
**Key files:** `src/analyst/api/{main,schemas,store}.py`, `app.py`,
`api_client.py`.
**Design decisions:** the API layer adds no business logic of its own —
routes validate, call into `analyst`, and shape the response; `Store` is an
explicit, swappable in-memory persistence seam (a dict behind a lock); the
UI never imports `analyst` — it only calls the API over HTTP through
`api_client.py` (enforced by a test, see
[Section 19](#section-19--testing-architecture)).
**Follow-up fixes in the same phase:**
- `ff84055` — the package wasn't actually pip-installed, so
  `uvicorn analyst.api:app` failed outside a dev session with `PYTHONPATH`
  set; fixed by declaring real dependencies in `pyproject.toml` and making
  `requirements.txt` install the project itself (`-e .[dev]`).
- `8c40327` — a Groq 429 (rate limit) was being reported to the client as a
  generic 502; `LLMProviderError` gained a `status_code` field so the API
  can return 429 with a friendly message instead.
- `2d6c562` — the UI now shows the actually-configured provider/model
  (read live from `/api/health`), rather than nothing.
**Verification:** `test_api.py` (47 tests), `test_ui.py` (23 tests).
**Resulting capability:** the full HTTP API listed in
[Section 13](#section-13--fastapi-backend), and a working Streamlit app.

### Phase 6.1 — Dockerization (`4bf96ae`)

**Objective:** containerize both services without changing their behavior.
**Key files:** `Dockerfile.api`, `Dockerfile.ui`, `docker-compose.yml`,
`.dockerignore`.
**Verification:** manually verified — both images built, the compose stack
started, `/api/health` returned 200, a real CSV upload and a real Groq
question were exercised through the running containers.

### Phase 6 — Production Readiness & Deployment (`af529ff`, `20a99b1`)

**Objective:** close the gaps between "runs on my machine" and "runs on a
hosting platform."
**Key changes:**
- Added `logging` to `src/analyst/api/main.py` (there was none before this
  commit) — unexpected exceptions and provider failures are now logged
  server-side (see [Section 22](#section-22--logging-and-observability)).
- Both Dockerfiles switched from exec-form to shell-form `CMD` so `$PORT`
  (injected by Render and similar platforms) is honored, falling back to
  8000/8501 for local Docker use.
- Added `render.yaml` (a Render Blueprint defining both services) and
  `LICENSE` (MIT).
**Resulting capability:** the exact configuration now deployed on Render.

### Final documentation pass (`db6886c`)

Rewrote `README.md` to reflect the live deployment (URLs, provider, test
count) rather than a "run it locally" description. This handoff document
(`docs/PROJECT_HANDOFF.md`) was added after that commit and is the first
change to touch this file.

---

## Section 3 — Current System Architecture

```
                              User (browser)
                                    │
                                    ▼
                    Streamlit UI — app.py  (Render: ai-data-analyst-ui)
                                    │
                                    │  HTTP, via api_client.py
                                    ▼
                 FastAPI backend — analyst.api.main  (Render: ai-data-analyst-api)
                                    │
                                    ▼
                    Conversation (session) / Agent loop
                         analyst.conversation / analyst.agent
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                              ▼
              LLM Provider                   Tool Dispatcher
        (Gemini or Groq, one            analyst.tools.run_tool
         LLMClient interface)          only registered tools,
                     │                  only declared arguments
                     ▼                              ▼
        Structured tool request  ─────▶   Analysis Engine
        (ToolRequest: call_tool /         analyst.analysis
         answer / clarification)          (pandas / NumPy)
                                                     │
                                                     ▼
                                            AnalysisResult
                                                     │
                                                     ▼
                                         Evidence (EvidenceStep)
                                                     │
                              ┌──────────────────────┴──────────────────────┐
                              ▼                                             ▼
                    Grounding check                              LLM writes final
                 analyst.grounding.check_grounding                answer (from the
              (flags unsupported figures,                       verified evidence
               does not invent a "fix")                           only, not the data)
                              │                                             │
                              └──────────────────────┬──────────────────────┘
                                                       ▼
                                                  Final Answer
                                            (Answer / AnswerResponse)
                                                       │
                                                       ▼
                                              back to FastAPI → UI
```

### Layer-by-layer

**Streamlit UI (`app.py`)**
- Responsibility: render the page, hold per-browser-session UI state
  (`st.session_state`), call the API, render the response.
- Input: file bytes from `st.file_uploader`, question text from
  `st.chat_input`.
- Output: rendered HTML/widgets in the browser.
- Key functions: `main`, `_sidebar`, `_load_dataset`, `_conversation_panel`,
  `_render_answer`.
- Failure behavior: any `ApiError` from `api_client.py` is caught and shown
  via `st.error(...)`; the page never crashes on a backend failure.

**API client (`api_client.py`)**
- Responsibility: the UI's only path to the backend — a thin `httpx`
  wrapper. See [Section 15](#section-15--api-client).

**FastAPI backend (`src/analyst/api/main.py`, `schemas.py`, `store.py`)**
- Responsibility: HTTP surface, request validation (Pydantic), routing to
  `analyst` functions, converting exceptions to HTTP responses, in-memory
  storage.
- Input: HTTP requests (multipart file upload, JSON bodies).
- Output: JSON responses (Pydantic `BaseModel`s), or a mapped error.
- Key classes/functions: `create_app`, `_build_router`,
  `_register_error_handlers`, `Store`.
- Failure behavior: see [Section 21](#section-21--error-handling) for the
  full map.

**Conversation / Agent (`conversation.py`, `agent.py`)**
- Responsibility: turn one question (plus optional prior turns) into a
  bounded sequence of tool calls and a final grounded answer.
- Input: a `DataFrame`, a question string, an `LLMClient`, optionally a
  `DatasetProfile` and bounded history.
- Output: an `Answer` (`agent.py`) dataclass.
- Key classes: `Conversation`, `Answer`, `EvidenceStep`.
- Failure behavior: `InvalidOperationError` for a bad question/empty
  dataset; `LLMProviderError`/`LLMConfigurationError` propagate up (they
  are not the agent's to fix); a tool failure is fed back to the model, not
  raised (see [Section 10](#section-10--agent-loop)).

**LLM Provider (`llm.py`, `gemini.py`, `groq.py`)**
- Responsibility: turn a prompt into text, behind one interface.
- Input: a prompt string, an optional system prompt, a `json_output` flag.
- Output: the model's reply as a string.
- Key classes: `LLMClient` (a `Protocol`), `GeminiClient`, `GroqClient`.
- Failure behavior: any SDK exception becomes `LLMProviderError` (with
  `status_code` if the SDK exposed one); an empty reply becomes
  `LLMResponseError`; a missing/bad key becomes `LLMConfigurationError`.

**Tool Dispatcher (`tools.py`)**
- Responsibility: parse the model's JSON reply into a `ToolRequest`,
  validate it against the fixed registry (`TOOLS`), and — only for
  `call_tool` — invoke the matching Phase 2 function.
- Input: raw text from the LLM (for parsing), or an already-parsed
  `ToolRequest` (for dispatch).
- Output: a `ToolRequest`, or an `AnalysisResult`.
- Failure behavior: `LLMResponseError` for unparseable JSON;
  `UnknownToolError` for a tool name not in `TOOLS`;
  `InvalidToolArgumentsError` for a missing/unknown/mistyped argument.

**Analysis Engine (`analysis.py`) / pandas / NumPy**
- Responsibility: the actual computation. See
  [Section 6](#section-6--analysis-engine).
- Failure behavior: `ColumnNotFoundError`, `InvalidOperationError`.

**Evidence / Grounding (`agent.py`'s `EvidenceStep`, `grounding.py`)**
- See [Section 12](#section-12--evidence-and-grounding).

---

## Section 4 — Repository Structure

Verified against the actual repository (`find` over the tracked tree,
`.venv`, `.git`, `__pycache__`, `.pytest_cache`, `data/` excluded):

```
AI-DATA-ANALYST-AGENT/
├── src/
│   └── analyst/
│       ├── __init__.py         # public API surface (re-exports)
│       ├── loader.py           # Phase 1 — file → DataFrame
│       ├── validation.py       # Phase 1 — file/frame/column checks
│       ├── profiling.py        # Phase 1 — DatasetProfile
│       ├── conversion.py       # numpy/pandas → plain Python
│       ├── errors.py           # every exception type in the project
│       ├── analysis.py         # Phase 2 — 7 deterministic operations
│       ├── charts.py           # Phase 2 — 6 Matplotlib chart builders
│       ├── context.py          # DatasetProfile → DatasetContext (prompt schema)
│       ├── tools.py            # tool registry, ToolRequest, dispatcher
│       ├── prompts.py          # the three prompts the agent uses
│       ├── llm.py              # LLMClient protocol, provider resolution
│       ├── gemini.py           # GeminiClient (only file importing google-genai)
│       ├── groq.py             # GroqClient (only file importing groq)
│       ├── agent.py            # the bounded agent loop
│       ├── conversation.py     # Conversation, Turn, bounded history
│       ├── grounding.py        # numeric grounding check
│       └── api/
│           ├── __init__.py     # exports `app`
│           ├── main.py         # FastAPI app factory, routes, error handling
│           ├── schemas.py      # Pydantic request/response models
│           └── store.py        # in-memory Store (datasets + sessions)
├── tests/                      # 342 tests, one file per module above
│   ├── conftest.py             # shared fixtures: sample_frame, analysis_frame,
│   │                           # FakeLLM, BrokenLLM
│   └── test_*.py               # see Section 19 for the full list
├── app.py                      # Streamlit UI entry point
├── api_client.py               # the UI's HTTP client for the API
├── data/
│   └── .gitkeep                # data/ itself is gitignored except this
├── Dockerfile.api               # FastAPI container
├── Dockerfile.ui                # Streamlit container
├── docker-compose.yml           # local two-service stack
├── .dockerignore
├── render.yaml                  # Render Blueprint (both services)
├── pyproject.toml               # dependencies, build config, pytest config
├── requirements.txt             # `-e .[dev]` — installs the project itself
├── .env.example                 # placeholder env vars, no real values
├── .gitignore
├── README.md                    # portfolio-facing documentation
├── LICENSE                      # MIT
└── docs/
    └── PROJECT_HANDOFF.md        # this document
```

Not documented in the table above because they are trivial or
self-explanatory: `__init__.py` files that only re-export names, and
`src/ai_data_analyst_agent.egg-info/` (a `pip install -e .` build artifact,
gitignored, regenerated automatically — not something to hand-edit or
commit).

---

## Section 5 — Data Ingestion and Profiling

**Supported formats:** CSV and XLSX only (`SUPPORTED_EXTENSIONS = (".csv",
".xlsx")` in `validation.py`). XLSX is read via `pandas.read_excel(...,
engine="openpyxl")`.

**Loading flow (`loader.py::load_dataset`):**
1. `validate_source(path)` — the path must exist, be a file, have a
   supported extension, and be non-zero bytes. Raises `DatasetNotFoundError`,
   `UnsupportedFileTypeError`, or `EmptyDatasetError` respectively.
2. `_read(source)` — dispatches to `pd.read_csv` or `pd.read_excel`.
   `pandas.errors.EmptyDataError` becomes `EmptyDatasetError`; any other
   parse failure (corrupt file, bad zip for XLSX, encoding issue) becomes
   `DatasetReadError`, with the original exception chained (`from exc`).
3. `validate_frame(frame, source)` — a frame with no columns or no rows
   (e.g. a header-only CSV) is rejected as `EmptyDatasetError`, even though
   step 1 already passed (the file itself was non-empty, but held no data).

**File size and temporary file behavior (in `src/analyst/api/main.py`, not
`loader.py` — the loader itself doesn't enforce a size limit):**
- The API layer caps uploads at `MAX_UPLOAD_BYTES = 10 * 1024 * 1024`
  (10 MB), checked in `_check_size` **after** the full upload is read into
  memory via `await file.read()`.
- `_parse_upload` writes the bytes to a `tempfile.NamedTemporaryFile` whose
  name is **generated by Python**, never derived from the client's
  filename, then calls `load_dataset` on that path. The temp file is
  deleted in a `finally` block (`Path(handle.name).unlink(missing_ok=True)`)
  regardless of whether parsing succeeded.
- The client's original filename is kept only for display, run through
  `_safe_name` (`Path(filename or "dataset").name[:120]`), which strips any
  directory components — it is never used to construct a filesystem path.

**Dataset validation summary:**

| Case | Where caught | Exception |
|---|---|---|
| Path doesn't exist / isn't a file | `validate_source` | `DatasetNotFoundError` |
| Wrong extension | `validate_source` | `UnsupportedFileTypeError` |
| Zero-byte file | `validate_source` | `EmptyDatasetError` |
| Corrupt/unparseable content | `_read` | `DatasetReadError` |
| Parses but has no rows/columns | `validate_frame` | `EmptyDatasetError` |
| Upload exceeds 10 MB | API layer, `_check_size` | `DatasetError` |

**Datatype classification (`profiling.py`):**
- `_is_numeric(series)`: numeric dtype **excluding** booleans.
- `_is_categorical(series)`: booleans, plus anything that is neither
  numeric nor a real datetime dtype (so text/object/category columns).
  Booleans are deliberately categorical, not numeric — confirmed by both
  `profiling.py` and `validation.py::require_numeric`, which explicitly
  rejects boolean columns even though pandas reports them as a numeric
  dtype internally.
- `_is_text(series)`: object or pandas' string dtype — the pool of
  candidates for datetime/numeric-like detection below.

**Datetime detection (`_detect_datetime_columns`):** A column already typed
as a real datetime dtype is reported directly. A **text** column is only
reported as a likely datetime when **both**:
1. At least `DETECTION_THRESHOLD = 0.9` (90%) of its non-null values parse
   as a date via `pd.to_datetime(..., errors="coerce")`, **and**
2. Fewer than 90% of those same values also parse as plain numbers via
   `pd.to_numeric(..., errors="coerce")`.

*Why conservative:* condition 2 exists specifically so a column of bare
years (`"2020"`, `"2021"`, ...) or numeric IDs is not mistaken for a date —
`pd.to_datetime` will happily parse a 4-digit string as a year, so without
the numeric-exclusion check, an ID column could be misreported as a date
column, which would then mislead both a human reading the profile and the
LLM choosing a `group_by_period` tool call against a non-date column.

**Why raw DataFrames are never passed to the LLM:** structurally enforced —
`context.py::build_context` takes a `DatasetProfile`, not a `DataFrame`, and
`DatasetContext.to_prompt_text()` only ever renders column names, dtypes,
missing counts, and (for categorical columns) one example value per column
— never a row of actual data. The agent (`agent.py`) only ever calls
`build_context(profile)`, never anything that touches `frame` directly for
prompt construction.

**`DatasetProfile` (dataclass, `profiling.py`):** `row_count`,
`column_count`, `column_names`, `dtypes`, `duplicate_row_count`, `columns`
(list of `ColumnProfile`), `numeric_columns` (list of `NumericStats`),
`categorical_columns` (list of `CategoricalStats`), `datetime_columns`
(list of `DatetimeColumn`), `warnings` (list of human-readable strings, e.g.
"3 duplicate row(s) found."). `.to_dict()` uses `dataclasses.asdict`, so the
whole structure is plain nested dicts/lists, JSON-serializable without any
custom encoder.

**JSON serialization:** every numeric/timestamp value that reaches a
`DatasetProfile`, `AnalysisResult`, or API response has passed through
`conversion.py::to_python` or `to_optional_float`, which convert numpy
scalars to native Python types, `pd.Timestamp` to an ISO string, and any
`NaN`/`NaT`/`None` to `None`.

---

## Section 6 — Analysis Engine

All seven operations live in `src/analyst/analysis.py`. Every one: (a)
validates its own arguments and raises `InvalidOperationError` or
`ColumnNotFoundError` on a problem, (b) never mutates the input `frame`,
(c) returns an `AnalysisResult`.

**`AnalysisResult`** (frozen dataclass): `operation` (str), `columns`
(list[str]), `parameters` (dict — what was asked for), `rows`
(list[dict] — the actual result rows, JSON-safe), `row_count` (int),
`metadata` (dict — operation-specific extras). `.to_dict()` for
serialization, `.to_frame()` rebuilds a `pandas.DataFrame` from `rows` (used
by `charts.py` and by the Streamlit UI's chart rendering).

**`Condition`** (frozen dataclass): `column`, `operator`, `value`. Built
either directly or via `Condition.from_mapping(dict)`. This is the *only*
way a filter is expressed — there is no expression-string parser anywhere
in this module.

| Operation | Purpose | Required args | Optional args | Notes |
|---|---|---|---|---|
| `filter_rows` | Keep rows matching all given conditions (AND) | `conditions: list[Condition\|dict]` | `limit: int` | Operators: `==`,`!=`,`>`,`>=`,`<`,`<=`,`in`,`not_in`. `in`/`not_in` require a list value. A type mismatch (e.g. comparing text `>` a number) raises `InvalidOperationError` rather than a raw pandas `TypeError`. |
| `sort_rows` | Order by one or more columns | `by: str\|list[str]` | `ascending: bool\|list[bool]` (default `True`), `limit: int` | Uses `kind="stable"` so ties keep their original relative order — results are reproducible across runs. |
| `group_aggregate` | Group by columns, aggregate others | `group_by: str\|list[str]`, `aggregations: dict[str, str\|list[str]]` | — | Allowed aggregations: `sum, mean, median, min, max, count`. Result columns are named `{column}_{aggregation}` (e.g. `revenue_sum`). `sum`/`mean`/`median` require the target column to be numeric (checked via `require_numeric`); `count`/`min`/`max` do not. |
| `describe_numeric` | count/mean/median/std/min/max | — | `columns: str\|list[str]` (default: every numeric column) | Raises `InvalidOperationError` if there are no numeric columns to describe. |
| `value_counts` | Frequency of each distinct value in one column | `column: str` | `limit: int` | Reports `count` and `percentage` per value, plus metadata: `distinct_values`, `non_null_count`, `missing_count`, `truncated`. Missing values are excluded from the counts. |
| `correlation` | Correlation matrix between numeric columns | — | `columns: list[str]` (default: all numeric), `method` (`pearson`\|`spearman`\|`kendall`, default `pearson`) | Requires at least 2 numeric columns. With exactly 2, the single pairwise value is duplicated into `metadata["correlation"]` for convenience. |
| `group_by_period` | Group rows by year or month of a date column | `datetime_column: str`, `period: "year"\|"month"` | `aggregations` (same shape as `group_aggregate`; omit to just count rows per period) | Parses the column on a **copy** — the caller's frame is never mutated, even if the column is stored as text. Rows that fail to parse as a date are dropped from the grouping and reported in `metadata["unparsed_rows"]`. |

**Result row limits:** `filter_rows`, `sort_rows`, and `value_counts` accept
an optional `limit`; when the result is truncated, `metadata["truncated"] =
True` and (for `filter_rows`/`sort_rows`) `metadata["matched_rows"]`/
`source_rows` tell the caller how many rows existed before truncation. This
row-level `limit` is separate from the evidence-level row cap the agent
applies before sending a result to the LLM (`MAX_EVIDENCE_ROWS = 50` in
`prompts.py` — see [Section 12](#section-12--evidence-and-grounding)).

**Timestamp handling:** any `pd.Timestamp` value that ends up in a result
row is converted to an ISO-8601 string by `conversion.to_python` before the
`AnalysisResult` is built (inside `_records`).

**Error handling summary:** `InvalidOperationError` for anything about the
*request* being invalid (bad operator, wrong aggregation name, non-numeric
column for a numeric-only aggregation, bad `limit`); `ColumnNotFoundError`
(raised by `validation.require_columns`) for any column name that isn't in
the frame — this is the exact mechanism that catches a model
"hallucinating" a column that doesn't exist.

**Why deterministic, not LLM-computed:** stated directly in the module
docstring — "callers choose the operation, pandas computes the answer." No
alternative code path exists in this codebase where a numeric result comes
from anywhere other than one of these seven pandas-backed functions;
allowing the LLM to compute arithmetic itself was never implemented, by
design (see [Section 20](#section-20--security-model) and
[Section 34](#section-34--interview--technical-discussion-notes)).

---

## Section 7 — Visualization Engine

All in `src/analyst/charts.py`. Every builder accepts either a raw
`pandas.DataFrame` or an `AnalysisResult` (via `_as_frame`, which calls
`.to_frame()` on an `AnalysisResult`) and returns a Matplotlib `Figure` —
nothing is written to disk, nothing is displayed automatically, and figures
are built with the object-oriented `Figure`/`Axes` API rather than
`pyplot`, so no global figure state accumulates across calls.

| Builder | Input | Output | Notes / limitations |
|---|---|---|---|
| `bar_chart(data, x, y)` | one categorical-ish `x`, one numeric `y` | `Figure` | `y` must be numeric; labels rotate 45° once there are more than 6 bars or a label exceeds 8 characters. |
| `line_chart(data, x, y)` | same shape as `bar_chart` | `Figure` | Same label-rotation rule. |
| `histogram(data, column, bins=10)` | one numeric column | `Figure` | `bins` must be a positive int; raises if the column has no non-null values. |
| `scatter_plot(data, x, y)` | two numeric columns | `Figure` | No trend line or regression — a plain scatter. |
| `box_plot(data, columns)` | one or more numeric columns | `Figure` | Every named column must have at least one non-null value. |
| `correlation_heatmap(data, columns=None, method="pearson")` | numeric columns (defaults to all) | `Figure` | Internally calls `analysis.correlation(...)` and renders its matrix — so it shares that function's "at least 2 numeric columns" requirement. |

`figure_to_png_bytes(figure, dpi=100)` renders any `Figure` to PNG bytes in
memory (`io.BytesIO`) — the intended path for an API response or a UI that
needs raw image bytes rather than an interactive figure object.

**How charts connect to analysis results / the UI.** This is an important,
easy-to-misread point: **`analyst.charts` is not wired into the agent or
the API as a callable tool.** It is a standalone library module, fully
tested (`test_charts.py`, 27 tests) and usable directly in Python
(`bar_chart(result, "region", "revenue_sum")`), but the LLM cannot request
a chart — it is not in the `TOOLS` registry in `tools.py`. The only chart
actually rendered end-to-end today is produced **client-side in
Streamlit**: `app.py::_maybe_chart` calls `st.bar_chart(...)` directly (a
Streamlit built-in, not `analyst.charts`) when a result's shape is
unambiguous — exactly one label column and one numeric column, between 2
and 50 rows. This is a deliberate, narrow heuristic, not a general chart
recommender.

---

## Section 8 — LLM Architecture

**`LLMClient`** (`llm.py`) — a `typing.Protocol`, `@runtime_checkable`. The
entire rest of the codebase depends on this interface and nothing more:

```python
class LLMClient(Protocol):
    def generate(
        self, prompt: str, *, system: str | None = None, json_output: bool = False
    ) -> str: ...
```

**`LLMConfig`** (frozen dataclass) — `api_key` (field with `repr=False`, so
it can never leak into a log line or an exception's default `repr`),
`model`, `provider`.

**`ProviderSettings`** (frozen dataclass) — `name`, `key_variable` (the env
var name holding the key), `model_variable` (the env var name holding the
model), `default_model`. `PROVIDER_SETTINGS` is a dict of these, one entry
per provider — this dict, plus one client class, is the entire surface a
new provider needs to touch (see
[Section 26](#section-26--how-to-add-a-new-llm-provider)).

**Provider resolution (`resolve_provider`):** `LLM_PROVIDER` wins if set
(case-insensitive); otherwise, whichever provider has a non-empty API key
wins, and Gemini wins if both keys are present. Raises
`LLMConfigurationError` naming the missing variables if neither key is set.

**`load_config(env=None, provider=...)`:** reads that one provider's key
and model variable. With `env=None` it loads `.env` (via `python-dotenv`,
if installed — a no-op otherwise) then reads `os.environ`; tests pass an
explicit `dict` instead, so they never touch the real environment.

**`create_client(provider=None, env=None)`:** the single entry point most
callers use — resolves the provider if not given explicitly, loads its
config, lazily imports and constructs the matching client class.

**Model configuration / temperature.** Both `GeminiClient` and `GroqClient`
hardcode `TEMPERATURE = 0.0` — not configurable via environment variable.
Rationale stated in both files: "Analysis should be reproducible, so ask
for the least creative output."

**Structured JSON response behavior:** when `json_output=True`,
`GeminiClient` sets `response_mime_type="application/json"` on the
`GenerateContentConfig`; `GroqClient` sets
`response_format={"type": "json_object"}` on the chat completion call. Both
are provider-native "force JSON" features. Even so, `tools.py::ToolRequest.
from_text` still parses defensively (strips code fences, falls back to
extracting the outermost `{...}` from surrounding prose) — the JSON mode is
a strong hint, not a guarantee, and the code does not assume the reply is
clean.

**Malformed output handling:** unparseable JSON, a JSON value that isn't an
object, a missing/invalid `action`, or a `call_tool` action with no tool
name all raise `LLMResponseError` from `ToolRequest.from_text`. Inside the
agent loop, this specific exception is caught and turned into a failed
`EvidenceStep` (`tool="(unreadable reply)"`) rather than aborting the whole
question — the model gets to try again within the step budget.

**Provider errors:** any exception from the underlying SDK call becomes
`LLMProviderError`. Both provider modules read the SDK's own numeric status
code if one is exposed — `.code` for `google-genai`'s exceptions, `.status_
code` for `groq`'s — and store it on `LLMProviderError.status_code`. Only
the exception's **type name** and that status code go into the message;
the original exception (which could contain request/response detail) is
chained via `from exc` for local debugging, but never serialized into a
message shown to a caller.

**Rate limiting:** a `status_code == 429` anywhere downstream is detected
specifically. In `src/analyst/api/main.py`'s error handler, a
`LLMProviderError` with `status_code == 429` is mapped to HTTP `429 Too
Many Requests` with a friendly message ("The model provider is
rate-limiting requests right now. Please try again shortly."); any other
`LLMError` becomes a generic `502 Bad Gateway`.

**Why provider-specific logic is isolated:** `gemini.py` and `groq.py` are,
by design and by comment in both files, "the only module that imports its
provider SDK." No other file in the project imports `google.genai` or
`groq` directly — so removing or breaking one provider's SDK dependency
cannot affect the other, and adding a third provider cannot require
touching the agent, the tool dispatcher, or the API layer.

**Environment variables involved (names only — see
[Section 18](#section-18--environment-configuration) for the full table):**
`LLM_PROVIDER`, `GOOGLE_API_KEY`, `MODEL_NAME`, `GROQ_API_KEY`,
`GROQ_MODEL_NAME`.

---

## Section 9 — Structured Tool Calling

**End-to-end example**, matching the actual live-verified interaction:

```
User: "What is the average salary by department?"
        │
        ▼
LLM replies (JSON, via build_agent_prompt / AGENT_RULES):
  {"action": "call_tool", "tool": "group_aggregate",
   "arguments": {"group_by": "department", "aggregations": {"salary": "mean"}}}
        │
        ▼
ToolRequest.from_text(reply)  →  ToolRequest(action="call_tool", tool="group_aggregate", ...)
        │
        ▼
tools.get_tool("group_aggregate")  →  looked up in the TOOLS dict (a fixed registry)
        │
        ▼
tools.validate_arguments(spec, arguments)  →  checks every arg against its declared
                                                type/required-ness/allowed values
        │
        ▼
tools.run_tool(frame, request)  →  calls analysis.group_aggregate(frame, **arguments)
        │
        ▼
AnalysisResult(operation="group_aggregate", rows=[{"department":"Engineering",
  "salary_mean": 93333.33}, {"department":"HR","salary_mean": 54000.0},
  {"department":"Marketing","salary_mean": 72000.0}], ...)
        │
        ▼
stored as an EvidenceStep, shown back to the LLM (build_answer_prompt / ANSWER_RULES)
        │
        ▼
LLM writes the final answer text using only those numbers
        │
        ▼
grounding.check_grounding(text, evidence)  →  GroundingReport(is_grounded=True, ...)
```

**`ToolRequest`** (`tools.py`, frozen dataclass): `action` (one of
`call_tool`, `answer`, `clarification`), `tool` (str, only for `call_tool`),
`arguments` (dict, default `{}`), `message` (str, only for `answer`/
`clarification`).

- `action = "call_tool"` — the model wants to run a named tool with the
  given arguments.
- `action = "answer"` — either the evidence already answers the question,
  or the dataset/tools genuinely cannot ("This dataset has no revenue
  column."). `message` becomes the answer text directly, with no further
  LLM call (see `agent.py`'s early return when `not any(step.succeeded ...)`).
- `action = "clarification"` — the question is ambiguous (a column, filter,
  measure or time range the model would have to guess). `message` is the
  one clarifying question shown to the user. This ends the loop
  immediately — no tool runs, and the caller (`Conversation`/API) surfaces
  `kind="clarification"`.

**Parsing (`ToolRequest.from_text`):** strips a ` ```json ` code fence if
present (`_strip_code_fence`); tries `json.loads` on the result; if that
fails, falls back to extracting the outermost `{...}` substring from the
text and parsing that (`_parse_embedded_object`) — handling the case where
the model wraps its JSON in a sentence ("Sure! Here you go: {...} Hope that
helps."). Any failure at any stage raises `LLMResponseError`.

**Tool registry (`TOOLS: dict[str, ToolSpec]`):** exactly seven entries,
one per Phase 2 operation, each a `ToolSpec(name, purpose, arguments,
function)` where `function` is a **direct reference** to the real
`analysis.py` function (e.g. `function=group_aggregate`) — there is no
re-implementation, no string-based lookup by name via `getattr`, and no
`eval`. `describe_tools()` returns the JSON-serializable description (name,
purpose, argument name/type/required/allowed_values) that goes into the
agent prompt; the `function` reference itself is never serialized or
exposed.

**Dispatcher (`run_tool`):** only accepts a `ToolRequest` with
`action == "call_tool"`. Looks the tool name up via `get_tool`, which
raises `UnknownToolError` for anything not a key in `TOOLS` — there is no
fallback, no partial match, no dynamic import based on the name string.
Arguments are validated by `validate_arguments` before the function is
ever called:
- Unknown argument name → `InvalidToolArgumentsError`.
- Missing required argument → `InvalidToolArgumentsError`.
- Wrong type (checked against `_TYPES`, a fixed
  JSON-type-name → Python-type mapping; booleans are explicitly excluded
  from numeric types since `bool` is a `int` subclass in Python) →
  `InvalidToolArgumentsError`.
- Value not in `allowed_values` (e.g. `period` must be `"year"` or
  `"month"`) → `InvalidToolArgumentsError`.

**Why arbitrary code execution is prohibited:** there is no code path in
`tools.py`, `agent.py`, or anywhere else that takes model-generated text
and passes it to `eval`, `exec`, `subprocess`, a dynamic `import`, or a
query-string interpreter. The only thing a model-generated string can ever
become is: (a) a lookup key into the fixed `TOOLS` dict, or (b) a value
passed as a typed keyword argument into one of seven known Python
functions. This was verified directly by searching the source tree for
`eval(`, `exec(`, `__import__`, `importlib`, `subprocess`, `os.system`,
`os.popen`, and `pickle.load` — none appear anywhere in `src/`, `app.py`,
or `api_client.py`.

---

## Section 10 — Agent Loop

This is implemented entirely in `src/analyst/agent.py`, function
`answer_question`. It is the single most important control-flow function in
the project.

**Constants:** `DEFAULT_MAX_STEPS = 5` — "Enough for a two or three step
analysis plus a correction, low enough to bound cost and latency" (module
comment). Callable with a different `max_steps` (`Conversation` also
exposes this), but the default is what production uses.

**Lifecycle of one call to `answer_question`:**

1. Validate inputs: non-empty question, non-empty frame, `max_steps` a
   positive int. Raise `InvalidOperationError` immediately if not — no LLM
   call happens for a request that's invalid on its face.
2. Build `context = build_context(profile or profile_dataset(frame))` —
   once, reused for every step of this question.
3. Loop `for number in range(1, max_steps + 1)`:
   a. `_next_action(...)` calls the LLM (`build_agent_prompt`, system =
      `AGENT_RULES`, `json_output=True`), passing the question, the schema,
      the tool descriptions, **every prior step this question** (including
      failures), and the bounded conversation `history`.
   b. If the reply doesn't parse (`LLMResponseError`), that's recorded as a
      failed `EvidenceStep` (`tool="(unreadable reply)"`) and the loop
      continues to the next iteration — the model is not told anything
      extra beyond seeing that failed step next time; it has to infer from
      the accumulated steps that its last reply didn't work.
   b. If `action == "clarification"` → return immediately, `kind=
      "clarification"`, no further LLM call.
   c. If `action != "call_tool"` (i.e. `"answer"`):
      - If no step has succeeded yet this question → return immediately;
        the model's own message is the answer (no grounding is computed,
        because there is no evidence to ground against).
      - If at least one step has succeeded → **break** out of the loop
        (the model has decided the evidence gathered so far is enough) and
        proceed to step 5.
   d. If `action == "call_tool"` → `_run_step` dispatches it. Success or
      failure, the resulting `EvidenceStep` is appended to `steps` and the
      loop continues to the next iteration (up to `max_steps`).
4. If the `for` loop exhausts all `max_steps` iterations without a `break`
   or early `return` (Python's `for...else`), return `kind="incomplete"`
   with a fixed message: *"I could not finish this analysis within the
   step limit. Try asking for one specific figure at a time."*
5. `_final_answer(...)`: collects every successful step's result as
   `evidence`, calls the LLM once more (`build_answer_prompt`, system =
   `ANSWER_RULES`, **no** `json_output` this time — plain text is expected)
   to write the final wording, then runs `check_grounding(text, evidence)`
   and returns the completed `Answer`.

**A concrete multi-step example** (hypothetical, but architecturally
accurate — mirrors what the live system did in Section 2's Phase 4/5
verification, extended to two tool calls):

```
Question: "Compare average salary and headcount by department."

Step 1 — LLM: {"action":"call_tool","tool":"group_aggregate",
               "arguments":{"group_by":"department","aggregations":{"salary":"mean"}}}
         → succeeds, EvidenceStep #1 recorded.

Step 2 — LLM (sees step 1's result in the prompt):
         {"action":"call_tool","tool":"group_aggregate",
          "arguments":{"group_by":"department","aggregations":{"department":"count"}}}
         → succeeds, EvidenceStep #2 recorded.

Step 3 — LLM: {"action":"answer","message":"Evidence gathered."}
         → at least one step succeeded, so the loop breaks here (not an
           early return) and proceeds to _final_answer.

Final answer call — LLM is shown both EvidenceStep results (not the raw
frame) and writes: "Engineering has the highest average salary at 93,333.33
across 3 employees; ..." grounded against both results.
```

**Termination guarantee:** the loop is a bounded `for` over
`range(1, max_steps + 1)` — there is no `while True`, no recursion, and no
code path that re-enters the loop with a fresh budget. Every iteration,
successful or not, consumes one unit of the budget (even a malformed-reply
`continue` still consumes the current `number` — the `for` variable still
advances). The loop **must** terminate within `max_steps` LLM calls (plus
exactly one more for the final answer, when the loop exits via `break` or
falls through to `_final_answer`).

---

## Section 11 — Conversation State

Implemented in `src/analyst/conversation.py`, class `Conversation`.

**Constructor fields:** `frame` (the `DataFrame`, held for the life of the
object), `client` (an `LLMClient`), `profile` (a `DatasetProfile`, computed
once via `profile_dataset(frame)` if not supplied), `max_steps` (default
`DEFAULT_MAX_STEPS`), `history_turns` (default `DEFAULT_HISTORY_TURNS = 3`),
`turns` (a plain Python `list[Turn]`, starts empty).

Constructor validation: raises `InvalidOperationError` for an empty frame
or a negative `history_turns`.

**`Turn`** (frozen dataclass): `question`, `answer` (the text), `kind`,
`results` (list of the successful `AnalysisResult` dicts from that turn —
`Answer.results`, a property on `agent.Answer`).

**`.ask(question)`:** calls `answer_question(frame, question, client,
profile=self.profile, max_steps=self.max_steps, history=self.history())`,
then appends a new `Turn` built from the returned `Answer`, then returns
that `Answer`.

**Bounded history (`.history()`):** this is deliberately bounded **twice**:
1. Only the **last `history_turns`** turns are ever included
   (`self.turns[-self.history_turns:]`) — older turns fall off entirely,
   are not summarized, and are not recoverable from the `Conversation`
   object once they scroll out of this window.
2. Of those included turns, only the **newest** keeps its full result rows
   (`detailed=True` for the last index in the slice); every older included
   turn is reduced to a **summary** — just `operation`, `columns`,
   `parameters`, `row_count` (`_SUMMARY_FIELDS`), with the actual numeric
   rows stripped out entirely.

This means a follow-up like "which one is highest?" can be answered by
reading the *previous* turn's full numbers (still present), while the
prompt cannot grow without bound as a conversation gets longer — an older
turn contributes only a few bytes of metadata, not its data.

**`.reset()`:** clears `self.turns`, keeping the dataset/profile/client —
used by the "Start a new session" button in the UI (which actually creates
a brand-new backend session rather than calling `.reset()` directly — see
[Section 14](#section-14--streamlit-ui)).

**Session lifetime / API restart behavior:** a `Conversation` is a plain
in-memory Python object. In the deployed system, it lives inside a
`StoredSession` inside the FastAPI process's `Store` (see
[Section 13](#section-13--fastapi-backend)) — there is no database and no
serialization to disk anywhere in this path. **If the API process
restarts** (a deploy, a crash, or — on Render's free tier — the service
sleeping and waking, which is a full process restart, not a pause), every
`Store`, therefore every `Conversation`, therefore every `Turn`, is lost.
The UI has no way to detect this proactively; the next request against a
now-unknown `session_id` will fail with `UnknownResource` → HTTP 404
("No session with id '...'"). This is a known, documented limitation (see
[Section 23](#section-23--known-limitations)), not a bug — the project
never implemented persistent session storage.

---

## Section 12 — Evidence and Grounding

**`EvidenceStep`** (`agent.py`, frozen dataclass): `number` (1-indexed step
number within the question), `tool` (the tool name, or the literal string
`"(unreadable reply)"` for a malformed LLM reply, or `"(none)"` if a
`ToolRequest` somehow had no tool name), `arguments` (dict, default `{}`),
`result` (the tool's `AnalysisResult.to_dict()`, trimmed — or `None` on
failure), `error` (the exception message — or `None` on success). Property
`.succeeded` is simply `self.result is not None`.

**Successful vs. failed results:** a successful step's `result` is the
literal dict form of the `AnalysisResult`, run through
`prompts.trim_result` first. A failed step's `error` carries the exact
exception message from whichever layer rejected it (`UnknownToolError`,
`InvalidToolArgumentsError`, `ColumnNotFoundError`,
`InvalidOperationError`, or `LLMResponseError` for an unparseable reply) —
this is what lets the model read *why* its previous attempt failed and
correct it on the next iteration.

**Result row limits at the evidence layer (`prompts.trim_result`,
`MAX_EVIDENCE_ROWS = 50`):** if a tool's result has more than 50 rows, only
the first 50 are kept in the evidence shown to the model, and
`result["rows_omitted"] = <count>` is added so the model (and, downstream,
the UI) can tell the result was truncated rather than complete. This is
separate from the per-operation `limit` argument described in
[Section 6](#section-6--analysis-engine) — a tool call without an explicit
`limit` that happens to return, say, 200 grouped rows would still be capped
to 50 rows of *evidence*, even though the full `AnalysisResult.row_count`
correctly still reports 200.

**Grounding extraction (`grounding.py::check_grounding`):**
1. `_collect_numbers(evidence)` recursively walks every dict/list/string in
   the evidence structure and extracts every number it finds, including
   numbers embedded inside strings (e.g. a `"period": "2024"` value counts
   as evidence for the figure 2024).
2. A regex (`_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"`) finds every number-looking
   substring in the model's final answer **text**.
3. Each found number is compared against the evidence set via `_supported`:
   an exact match (within floating-point tolerance `1e-9`), **or** a
   rounding match — if the answer wrote a number with fewer decimal places
   than a known evidence value, rounding the evidence value to that many
   places and comparing is allowed (so an evidence value of `209.153` can
   legitimately be reported as `"209.15"`, `"209.2"`, or `"209"` — all
   count as grounded — but `"210"` does not, because no rounding of
   `209.153` to any precision produces `210`).
4. The literal values `0`, `1`, `2` are ignored entirely (`_IGNORED`) —
   these are almost always ordinals or sentence-structure numbers ("the
   two departments..."), not claims about the data, and checking them
   produced noise rather than signal.

**Output — `GroundingReport`** (frozen dataclass): `is_grounded` (bool —
`True` iff `unsupported` is empty), `checked` (every number-string that was
examined), `unsupported` (the subset that could not be matched to
evidence).

**Advisory, not enforced:** `check_grounding` is called once, after the
final answer text already exists, and its result is attached to the
`Answer`/`AnswerResponse` for the caller to see — nothing in the codebase
re-prompts the model, discards, or edits the answer based on the grounding
result. The module's own docstring states this directly: "This is a safety
net, not a proof... It cannot judge whether the wording of an answer is a
fair reading of the data." A caller (the UI, or a future integration) is
expected to read `grounding.is_grounded` and decide what to do with an
ungrounded answer; today, `app.py::_render_details` shows a `st.warning`
listing the unsupported figures when `is_grounded` is `False`, but still
displays the answer text itself.

**Why it exists:** it is the project's only automated check that the LLM's
*final wording* — which is free-form generated text, not a structured tool
call — didn't silently introduce, mis-transcribe, or invent a number while
paraphrasing a correct `AnalysisResult`. Every number up to that point (in
`analysis.py`, in `EvidenceStep.result`) is already guaranteed correct by
construction (pandas computed it); grounding is the check on the one step
where a plain-language rewrite could still drift from the underlying
number.

**Verified example** (from a real API response captured during end-to-end
testing, values abbreviated):

```json
{
  "answer": "The average salary is 79,666.67.",
  "result": {"rows": [{"column": "salary", "mean": 79666.67, ...}]},
  "grounding": {"is_grounded": true, "checked": ["79,666.67"], "unsupported": []}
}
```

---

## Section 13 — FastAPI Backend

`src/analyst/api/main.py` defines `create_app(...)`; the module-level
`app = create_app()` is what `uvicorn analyst.api:app` serves. All routes
are under the `/api` prefix.

**App factory (`create_app`)** accepts, all optional: `client_factory`
(defaults to `create_client` from `llm.py`; called **inside a handler**,
never as a FastAPI dependency, specifically so a malformed request body is
reported as `422` before any provider is touched), `provider_info`
(defaults to `_active_provider`, which reads `resolve_provider`/
`load_config` — configuration only, never builds an actual client or
contacts a provider), `store` (defaults to a new `Store()`), `allowed_
origins` (defaults to `_configured_origins()`, reading `CORS_ORIGINS`).
Tests inject their own `client_factory`/`provider_info`/`store` so they
never depend on the real environment or a real provider.

**Dependency injection:** `get_store(request)` and `get_provider_info
(request)` pull `app.state.store`/`app.state.provider_info` via FastAPI's
`Depends(...)`. `build_client(request)` is a plain function (not a FastAPI
dependency) called explicitly inside the `analyze`/`create_session`
handlers, for the ordering reason above.

**CORS:** `CORSMiddleware` with `allow_credentials=False`,
`allow_methods=["GET","POST","DELETE"]`, `allow_headers=["*"]`. Default
origins: `http://localhost:8501`, `http://127.0.0.1:8501` — overridable via
`CORS_ORIGINS` (comma-separated). Note: the browser talks to Streamlit, not
directly to this API (see [Section 14](#section-14--streamlit-ui)) — CORS
mainly matters for someone hitting `/docs`'s "Try it out" or a raw `curl`/
JS client from a different origin, not for the deployed UI's own traffic.

**Endpoints:**

| Method & Path | Purpose | Request body | Success response | Key behavior |
|---|---|---|---|---|
| `GET /api/health` | Liveness + active provider/model | — | `HealthResponse` (`status`, `version`, `provider`, `model`) | Never builds a client; safe to call with no provider configured (`provider`/`model` come back `null`). |
| `POST /api/datasets` | Upload a CSV/XLSX, get its profile | multipart `file` | `201`, `DatasetSummary` | Validates extension + size, writes+deletes a temp file, profiles it, stores it. |
| `GET /api/datasets/{dataset_id}` | Re-fetch a dataset's profile | — | `DatasetSummary` | `404` if unknown. Never returns rows. |
| `POST /api/analyze` | One-off question, no persisted session | `{dataset_id, question}` | `AnswerResponse` | Builds a client, calls `answer_question` directly (bypasses `Conversation`, so no follow-up history). |
| `POST /api/sessions` | Open a conversation over a dataset | `{dataset_id}` | `201`, `SessionResponse` | Checks the dataset exists (→ `404` if not) **before** building a client, so a bad `dataset_id` never triggers a provider call. |
| `GET /api/sessions/{session_id}` | Session state | — | `SessionResponse` (`turn_count`, etc.) | `404` if unknown. |
| `POST /api/sessions/{session_id}/questions` | Ask, with follow-up context | `{question}` | `AnswerResponse` | Calls `session.conversation.ask(...)` — this is the one path that actually uses bounded history. |
| `DELETE /api/sessions/{session_id}` | End a session | — | `204` | `404` if unknown. |

**Pydantic schemas (`schemas.py`):** `HealthResponse`, `ColumnInfo`,
`DatasetSummary`, `SessionRequest`/`SessionResponse`, `QuestionRequest`
(question capped at `MAX_QUESTION_LENGTH = 1000` chars, `min_length=1`),
`AnalyzeRequest` (extends `QuestionRequest` with `dataset_id`), `StepInfo`,
`GroundingInfo`, `AnswerResponse`, `ErrorResponse`. Converter functions
(`dataset_summary`, `session_response`, `answer_response`) build these from
the internal `StoredDataset`/`StoredSession`/`Answer` objects — the
internal dataclasses are never returned directly, so the API response
shape is decoupled from internal representations.

**`Store` (`store.py`):** a single in-process class holding two plain
dicts (`_datasets`, `_sessions`, keyed by a `uuid4().hex` id) behind one
`threading.Lock` (needed because FastAPI runs sync route handlers on a
thread pool). `add_dataset` profiles the frame once at upload time and
caches the `DatasetProfile` on the `StoredDataset`, so it is never
recomputed on later requests against the same dataset. No eviction policy
exists — see [Section 23](#section-23--known-limitations).

**Upload handling:** see [Section 5](#section-5--data-ingestion-and-profiling)
for the full validate → temp-file → parse → delete flow.

**Error handling / provider error mapping / rate-limit behavior:** see
[Section 21](#section-21--error-handling) for the complete map.

**Logging:** see [Section 22](#section-22--logging-and-observability).

---

## Section 14 — Streamlit UI

`app.py`, single-file, ~250 lines. State lives in `st.session_state`:
`api_url`, `dataset` (the last `DatasetSummary` dict), `session_id`,
`history` (list of `{question, answer}` for the current page session),
`uploaded_name`.

**Flow:**

1. **API URL configuration** — a text input in the sidebar, prefilled from
   `api_base_url()` (`api_client.py`), which reads `ANALYST_API_URL` from
   the environment. Editable at runtime without restarting the app.
2. **Dataset upload** — `st.file_uploader` (CSV/XLSX only). On a new file,
   `_load_dataset` uploads the bytes via `api.upload_dataset(...)`; on
   success it stores the returned `DatasetSummary` in session state, clears
   `history`, and immediately opens a new backend session
   (`_start_session`).
3. **Dataset profile** — `_dataset_panel` renders three `st.metric`s (row
   count, column count, "columns with gaps") and a table of every column's
   name/dtype/kind/missing/unique counts, plus an expander listing any
   profiling warnings.
4. **Session creation** — happens automatically after upload, and again if
   the user clicks "Start a new session" (which calls `_start_session`
   again against the *same* `dataset_id`, producing a **new**
   `session_id` — this is a fresh backend `Conversation`, not a call to
   `Conversation.reset()`, since the UI has no direct access to the backend
   object; it goes through the API and gets a brand-new session).
5. **Question submission** — `st.chat_input`. Requires an active
   `session_id` (shows `st.error` otherwise). Calls
   `api.ask(session_id, question)`.
6. **Answer rendering** (`_render_answer`) — the answer text; if
   `kind != "answer"` (i.e. `clarification` or `incomplete`), a
   `st.warning` banner with an icon/label from `KIND_LABELS` is shown above
   the text.
7. **Result table** — if the response's `result.rows` is non-empty, shown
   as an `st.dataframe`.
8. **Chart** — `_maybe_chart` only calls `st.bar_chart` when the result
   frame has exactly one non-numeric column and one numeric column, and
   between 2 and 50 rows — otherwise no chart is drawn (this is
   Streamlit's own `st.bar_chart`, not `analyst.charts` — see
   [Section 7](#section-7--visualization-engine)).
9. **Evidence/tool steps** — `_render_details`, inside an
   `st.expander("How this was worked out")`: each step's tool name, row
   count (or its error), and its arguments (`st.json`, collapsed).
10. **Grounding information** — inside the same expander, an `st.warning`
    listing any unsupported figures, only when `grounding.is_grounded` is
    `False`.
11. **Follow-up** — simply another message in the same `st.chat_input`
    loop, sent to the same `session_id`; the backend `Conversation`
    supplies the bounded history automatically (the UI does not construct
    or send any history itself).
12. **Reset** — the "Start a new session" button in the sidebar.

**Relationship between `app.py` and `api_client.py`:** `app.py` imports
`AnalystApi`, `ApiError`, `api_base_url` from `api_client.py` and calls
nothing else outside the Streamlit library and `pandas`. `app.py` contains
**zero** imports from the `analyst` package — verified both by direct
inspection and by an actual test
(`tests/test_ui.py::test_the_streamlit_app_never_imports_the_analysis_
engine`, which asserts `"from analyst"` and `"import analyst"` do not
appear in `app.py`'s source). The UI reaches the backend purely over HTTP.

---

## Section 15 — API Client

`api_client.py` — a single class, `AnalystApi`, wrapping `httpx.request`.

**Methods:** `health()`, `upload_dataset(filename, payload)`,
`dataset(dataset_id)`, `create_session(dataset_id)`,
`ask(session_id, question)`, `end_session(session_id)` — each a thin
one-line call into the shared `_request(method, path, **kwargs)`.

**Base URL:** `AnalystApi(base_url=None, timeout=120.0)`. If `base_url` is
not given, `api_base_url()` is used, which reads `ANALYST_API_URL` from the
environment, falling back to `DEFAULT_BASE_URL = "http://127.0.0.1:8000"`.
The constructor strips a trailing slash from whatever URL it's given.

**Timeout behavior:** every request uses the instance's `timeout` (default
120 seconds) — generous, because an LLM round trip (especially a
multi-step agent question) can legitimately take tens of seconds.

**Error handling:** `httpx.HTTPError` (connection refused, DNS failure,
timeout) is caught and re-raised as `ApiError("Could not reach the analyst
API at {base_url}. Is the backend running?")`. Any HTTP response with
`status_code >= 400` is turned into `ApiError(_error_message(response),
response.status_code)`. `_error_message` reads the API's own
`{"detail": ..., "code": ...}` shape (see `ErrorResponse` in
[Section 13](#section-13--fastapi-backend)); for a FastAPI validation error
(`detail` is a list, not a string), it extracts and formats the first
problem, e.g. `"question: Field required"`. A `204` or empty body returns
`{}`.

**Local vs. deployed API URL:** locally, `ANALYST_API_URL` is typically
unset (defaults to `http://127.0.0.1:8000`) or set to
`http://api:8000` when the UI itself is running inside Docker Compose
(the compose service name, not `localhost` — see
[Section 16](#section-16--docker-architecture)). In the deployed Render
configuration, `ANALYST_API_URL` on the `ai-data-analyst-ui` service is set
to the **public HTTPS URL** of the `ai-data-analyst-api` service
(`https://ai-data-analyst-api-8b3n.onrender.com`) — the two services do not
share a private network path in the current Render Blueprint (`render.
yaml` uses `sync: false` placeholders for this, filled in manually after
the API service is deployed; see
[Section 17](#section-17--render-deployment)).

---

## Section 16 — Docker Architecture

Two independent images, built from the repository root with the whole
project as build context; `.dockerignore` excludes `.git`, the virtualenv,
`__pycache__`/`*.egg-info`, `tests/`, `.env`/`.env.example`, `data/`,
editor directories, and misc log/OS files.

**`Dockerfile.api`:**
- Base image: `python:3.12-slim`.
- `ENV`: `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`,
  `PIP_NO_CACHE_DIR=1`, `PIP_DISABLE_PIP_VERSION_CHECK=1`.
- `WORKDIR /app`. Copies `pyproject.toml`, `README.md` (needed only because
  `pyproject.toml` references it as the long description), and `src/`.
- `RUN pip install --no-cache-dir .` — installs the project **non-editable**
  (unlike local dev's `-e .[dev]`), and deliberately **without** the `dev`
  extra (`pytest`/`httpx2`), so the runtime image doesn't carry test-only
  dependencies.
- Creates a non-root user `appuser` (uid 1000), `chown -R`s `/app`, then
  `USER appuser` for everything after.
- `EXPOSE 8000`.
- `CMD uvicorn analyst.api:app --host 0.0.0.0 --port ${PORT:-8000}` — shell
  form specifically so `${PORT:-8000}` is expanded by the shell at
  container start, not baked in at build time. `$PORT` is what Render (and
  similar platforms) injects; local `docker run`/`docker compose` don't set
  it, so it falls back to `8000`.

**`Dockerfile.ui`:** same base image and `ENV` block. Additionally copies
`app.py` and `api_client.py` (the two files the API image doesn't need).
Same non-root-user pattern. `EXPOSE 8501`. `CMD streamlit run app.py
--server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true
--browser.gatherUsageStats=false` — `--server.headless` and
`--browser.gatherUsageStats=false` specifically prevent Streamlit's
first-run browser-open attempt and telemetry prompt from hanging a
non-interactive container.

Both images reuse the **same** `pyproject.toml` dependency list (there is
no separate, hand-maintained dependency list for the UI image) — a
documented trade-off: the UI image ends up carrying packages it never
imports at runtime (FastAPI, Matplotlib, both provider SDKs), in exchange
for a single source of truth for dependencies rather than a second list
that could drift out of sync.

**`docker-compose.yml`:** two services, `api` and `ui`, no other
infrastructure (no database, no cache, no reverse proxy).
- `api`: builds `Dockerfile.api`; host port `${API_HOST_PORT:-8000}` mapped
  to container port 8000 (host side configurable specifically to dodge a
  port already in use on the developer's machine — the container's
  internal port is always 8000, which is what `ui` actually talks to);
  `env_file: [{path: .env, required: false}]` — so a clean checkout with no
  `.env` still builds and starts (just unconfigured), rather than failing
  to even come up; `environment: CORS_ORIGINS=http://localhost:8501,
  http://127.0.0.1:8501`; a `healthcheck` running `python -c
  "urllib.request.urlopen('http://localhost:8000/api/health', timeout=3)"`
  every 10s (5 retries, 10s start grace period).
- `ui`: builds `Dockerfile.ui`; host+container port both `8501`;
  `environment: ANALYST_API_URL=http://api:8000`; `depends_on: {api:
  {condition: service_healthy}}` — Compose will not start the `ui`
  container until the `api` container's healthcheck reports healthy.

**Why `ui` uses the Compose service name, not `localhost`, inside Docker:**
`localhost`/`127.0.0.1` inside a container refers to *that container's own*
network namespace, not the host machine or a sibling container. Docker
Compose's default bridge network gives each service a DNS name equal to its
service key (`api`), resolvable from any other service on the same network
— so `http://api:8000` is the only address from which the `ui` container
can actually reach the `api` container. This is called out explicitly in
both the compose file's own comment and the README.

**Known Docker limitations (verified/observed during manual testing, not
purely theoretical):**
- No `.env` in the repository means `docker compose up` on a clean
  checkout starts an **unconfigured** API (health check still passes; any
  question will fail with `LLMConfigurationError` → 503) until a real
  `.env` is added.
- If a host already has something listening on port 8000 (observed during
  development — an unrelated local service occupied it), the fix is the
  `API_HOST_PORT` override, not a change to the compose file's internal
  container port.
- The two Dockerfiles' shell-form `CMD` means the process started is a
  child of `/bin/sh -c "..."` — standard Docker behavior for shell-form
  CMD, and not observed to cause any problem in practice for this project
  (signal handling on `docker stop` has not surfaced an issue in testing),
  but worth knowing if a future change needs precise PID-1 signal
  semantics.

---

## Section 17 — Render Deployment

**Services** (both defined in `render.yaml`, a Render "Blueprint"):

| Service name | Type | Dockerfile | Plan | Health check path |
|---|---|---|---|---|
| `ai-data-analyst-api` | `web`, `env: docker` | `./Dockerfile.api` | `free` | `/api/health` |
| `ai-data-analyst-ui` | `web`, `env: docker` | `./Dockerfile.ui` | `free` | *(none configured)* |

Both use `dockerContext: .` (repo root), matching local Docker builds
exactly — the same two Dockerfiles, no Render-specific build steps.

**Actual live URLs (verified 2026-09-10):**
- API: `https://ai-data-analyst-api-8b3n.onrender.com`
- UI: `https://ai-data-analyst-ui-tfvu.onrender.com`

*(The exact subdomain suffix — `-8b3n`, `-tfvu` — is assigned by Render at
service creation and is not predictable or reproducible from `render.yaml`
alone; a fresh deploy from the Blueprint will get different suffixes. See
[Section 28](#section-28--how-to-redeploy-from-scratch).)*

**Environment variables on the API service** (all `sync: false` in
`render.yaml`, meaning: not stored in the file, entered by hand in the
Render dashboard): `LLM_PROVIDER`, `GOOGLE_API_KEY`, `MODEL_NAME`,
`GROQ_API_KEY`, `GROQ_MODEL_NAME`, `CORS_ORIGINS` (optional). **Verified
current values:** `LLM_PROVIDER=groq` (or `GROQ_API_KEY` alone with no
`GOOGLE_API_KEY`, since Groq is the one active — the exact mechanism
`resolve_provider` uses is documented in
[Section 8](#section-8--llm-architecture); which of the two applies here is
**not verified from repository** since it's a dashboard setting, not
source-controlled — either configuration produces the observed
`"provider": "groq"` health response).

**Environment variable on the UI service:** `ANALYST_API_URL`, set to the
API service's own public URL (`https://ai-data-analyst-api-8b3n.
onrender.com`).

**`$PORT`:** Render sets `PORT` on both services; both Dockerfiles' `CMD`
reads it via `${PORT:-...}` (see [Section 16](#section-16--docker-architecture)).
This is **not** a project-specific Render setting — it is Render's standard
mechanism for any web service, and the Dockerfiles were specifically
updated (commit `af529ff`) to honor it.

**Health check:** Render polls `healthCheckPath: /api/health` on the API
service to decide whether a deploy is healthy before routing traffic to it.
The UI service has no `healthCheckPath` configured in `render.yaml` — Render
falls back to its default behavior of checking that the service is
listening on `$PORT` (Streamlit's root path, since no path is specified —
not independently verified against Render's exact default beyond what a
successful deploy demonstrates in practice).

**Service relationship:** the two services are **independent** Render web
services with **no** private networking configured between them in the
current Blueprint — the UI reaches the API over the public internet via
HTTPS, the same as any external client would. This is architecturally
identical to how a browser reaches the UI itself; there is no
Render-internal shortcut in use here (unlike Docker Compose's private
bridge network locally).

**Step-by-step deployment procedure** (this is what was actually followed,
reconstructed from the Blueprint structure and the verified end state —
see [Section 28](#section-28--how-to-redeploy-from-scratch) for the fully
generalized version to use if starting over):

1. Connect the GitHub repository to a Render account.
2. Create a new Blueprint from the repository; Render reads `render.yaml`
   and proposes both services.
3. Configure the API service's environment variables (provider, key,
   model) in the Render dashboard.
4. Deploy the API service.
5. Verify `GET https://<api-service>.onrender.com/api/health` returns
   `200` with the expected provider/model.
6. Note the API service's public URL.
7. Configure the UI service's `ANALYST_API_URL` to that URL.
8. Deploy the UI service.
9. Verify the UI loads and shows the correct "AI Model: ..." caption
   (sourced live from `/api/health` — see
   [Section 14](#section-14--streamlit-ui)).
10. Run an end-to-end test: upload a dataset, ask a question, verify a
    real answer comes back with grounded evidence. **This exact procedure
    was verified** — dataset upload, group aggregation, a real Groq
    inference, and a follow-up question all succeeded on the live
    deployment (see [Section 2](#section-2--project-history--development-roadmap),
    "current known project state" figures in the task that produced this
    document, and [Section 31](#section-31--verified-production-checklist)).

---

## Section 18 — Environment Configuration

Every environment variable this project reads anywhere, verified by
grepping the actual source for each name. **No real values appear below —
only variable names and placeholders.**

| Variable | Required? | Placeholder example | Read by | Purpose |
|---|---|---|---|---|
| `LLM_PROVIDER` | No | `LLM_PROVIDER=groq` | `llm.py::resolve_provider` | `"gemini"` or `"groq"`; if unset, whichever key is present wins (Gemini if both). |
| `GOOGLE_API_KEY` | Only if using Gemini | `GOOGLE_API_KEY=<your-key>` | `llm.py::load_config` (via `PROVIDER_SETTINGS[GEMINI]`) | Gemini credentials. |
| `MODEL_NAME` | No | `MODEL_NAME=gemini-2.5-flash` | `llm.py::load_config` | Gemini model; defaults to `gemini-2.5-flash` (`DEFAULT_MODEL`). |
| `GROQ_API_KEY` | Only if using Groq | `GROQ_API_KEY=<your-key>` | `llm.py::load_config` (via `PROVIDER_SETTINGS[GROQ]`) | Groq credentials. **Currently the active provider in production.** |
| `GROQ_MODEL_NAME` | No | `GROQ_MODEL_NAME=openai/gpt-oss-120b` | `llm.py::load_config` | Groq model; defaults to `llama-3.3-70b-versatile` (`DEFAULT_GROQ_MODEL`). **Production is currently set to `openai/gpt-oss-120b`, overriding the default.** |
| `CORS_ORIGINS` | No | `CORS_ORIGINS=https://ai-data-analyst-ui-tfvu.onrender.com` | `src/analyst/api/main.py::_configured_origins` | Comma-separated list of origins allowed to call the API directly from a browser. Defaults to the two local Streamlit URLs. |
| `ANALYST_API_URL` | No | `ANALYST_API_URL=http://127.0.0.1:8000` | `api_client.py::api_base_url` | Where the Streamlit UI looks for the backend. Defaults to `http://127.0.0.1:8000`. |
| `PORT` | No (platform-set) | `PORT=10000` | `Dockerfile.api` / `Dockerfile.ui` `CMD` (shell expansion, not Python code) | Overrides the container's listen port. Render sets this automatically; falls back to 8000 (API) / 8501 (UI) if unset. |

**Two variables that exist in `.env.example` but are *not* read by any
code in this repository:** `API_HOST` and `API_PORT`. They appear as
documentation/placeholder guidance for someone manually invoking `uvicorn
--host $API_HOST --port $API_PORT`, but no Python source file, Dockerfile,
or compose file references either name. Verified directly (`grep -rn
"API_HOST\|API_PORT" src/ app.py api_client.py` returns no matches outside
`.env.example`). Do not assume setting these has any effect — they are
informational only.

**Provider-specific configuration:** to run with **Gemini**, set
`GOOGLE_API_KEY` (and optionally `LLM_PROVIDER=gemini`, `MODEL_NAME`). To
run with **Groq**, set `GROQ_API_KEY` (and optionally
`LLM_PROVIDER=groq`, `GROQ_MODEL_NAME`). Setting both keys without
`LLM_PROVIDER` will silently prefer Gemini (`resolve_provider`'s tie-break
rule) — set `LLM_PROVIDER` explicitly if that's not the intent.

**Where these are supplied in each environment:**

| Environment | Mechanism |
|---|---|
| Local (no Docker) | A `.env` file at the repo root, loaded by `python-dotenv` when `load_config`/`resolve_provider` is called with no explicit `env` argument. |
| Local Docker Compose | The same `.env` file, loaded by Compose's `env_file: [{path: .env, required: false}]` on the `api` service only — `ui` gets `ANALYST_API_URL` from `docker-compose.yml`'s own `environment:` block, not from `.env`. |
| Render | Set per-service in the Render dashboard's Environment tab; `render.yaml` only declares the variable *names* (`sync: false`), never values. |
| Tests | Never read from the real environment — `load_config`/`resolve_provider` are always called with an explicit `dict` in tests, and the FastAPI test app is built with an injected `client_factory`/`provider_info` (see [Section 19](#section-19--testing-architecture)). |

---

## Section 19 — Testing Architecture

**Command:** `pytest` from the repo root (after `pip install -r
requirements.txt`, which installs the `dev` extra). `pyproject.toml`
configures `testpaths = ["tests"]`, `pythonpath = ["src", "."]`, and
`filterwarnings = ["error", ...]` with a short allow-list of specific
third-party deprecation warnings (from `starlette`'s test client and
`anyio`) — meaning **any warning from this project's own code fails the
test suite**, not just an assertion failure.

**Verified current status:** `342 passed` (re-run and confirmed while
producing this document; also independently listed per file below via
`pytest --collect-only`).

**Test files and counts (verified by direct collection, not estimated):**

| File | Count | Covers |
|---|---|---|
| `test_agent.py` | 30 | The agent loop: one-tool and multi-tool questions, the step limit, malformed replies, unknown tools, bad arguments, hallucinated columns, clarification, unsupported requests, provider failures, empty datasets/questions. |
| `test_analysis.py` | 58 | All seven `analysis.py` operations, `Condition`, result shape/serialization. |
| `test_api.py` | 47 | Every HTTP endpoint, upload validation, error-code mapping (400/404/422/429/500/502/503), rate-limit mapping, secret-leak checks, CORS. |
| `test_charts.py` | 27 | All six chart builders, `AnalysisResult`-as-input, error cases (empty data, non-numeric columns). |
| `test_conversation.py` | 21 | `Conversation`, bounded history (turn limit and row-summarization), `reset()`, cross-provider compatibility, grounding on real turns. |
| `test_gemini.py` | 15 | `GeminiClient` against a stubbed SDK — no real API key or network call. |
| `test_llm_setup.py` | 22 | `DatasetContext`/`build_context`, prompt builders, configuration loading. |
| `test_loader.py` | 12 | `load_dataset` — valid/invalid files, all `DatasetError` subclasses. |
| `test_profiling.py` | 17 | `profile_dataset` — dtypes, missing values, duplicates, datetime detection. |
| `test_providers.py` | 34 | Provider selection/resolution, `GroqClient` against a stubbed SDK, `create_client`. |
| `test_smoke.py` | 1 | The package imports at all. |
| `test_tools.py` | 35 | The tool registry, `ToolRequest` parsing, the dispatcher, argument validation. |
| `test_ui.py` | 23 | `api_client.py`'s HTTP wrapper, and that `app.py` never imports `analyst` directly. |
| **Total** | **342** | |

**Fixtures (`tests/conftest.py`):** `sample_frame` (6 rows, deliberately
carrying missing values, a duplicate row, and text-stored dates — used by
Phase 1 tests); `analysis_frame` (6 sales rows where `revenue == units *
10`, so correlation is exactly 1.0 — used by Phase 2+ tests);
`csv_file`/`xlsx_file` (materialize `sample_frame` to a temp path);
`FakeLLM` (a scripted `LLMClient` — returns queued replies in order,
records every call's prompt/system/json_output for assertions); `BrokenLLM`
(always raises, for provider-failure tests). **No test in the suite
constructs a real `GeminiClient`/`GroqClient` against the live network** —
provider tests stub the SDK import point (`analyst.gemini._import_sdk` /
`analyst.groq._import_sdk`) via `monkeypatch`.

**API tests specifically** use `fastapi.testclient.TestClient` against an
app built with `create_app(client_factory=..., provider_info=...,
store=Store())` — always an injected fake, never `create_client` for real,
and always a fresh `Store()` per test so tests can't see each other's
datasets/sessions.

**Security-related checks embedded in the test suite** (not a separate
file — spread across the relevant modules' tests): no key/secret substring
ever appears in an API response body (`test_api.py`); a provider exception
carrying a fake secret string never reaches the raised
`LLMProviderError`'s message (`test_gemini.py`, `test_providers.py`); a
malicious/traversal-style filename (`"../../evil.csv"`) is reduced to a
safe basename, never used as a path (`test_api.py`).

---

## Section 20 — Security Model

**No arbitrary code execution — verified directly, not assumed.** A search
of the entire source tree (`src/`, `app.py`, `api_client.py`) for `eval(`,
`exec(`, `__import__`, `importlib`, `subprocess`, `os.system`, `os.popen`,
and `pickle.load` returns **zero matches**. The only "dynamic" behavior
anywhere near model output is a fixed-key dictionary lookup
(`TOOLS[name]`) and lazy `import` statements for the **provider SDKs**
(`from google.genai import ...` / `import groq`) — both of which are hard
-coded module paths in `gemini.py`/`groq.py`, never constructed from a
string the model supplied.

**Explicit tool registry:** the model's only lever on the data is
`tools.TOOLS`, a fixed dict of exactly seven entries, each pointing
directly at a real `analysis.py` function. An unregistered tool name raises
`UnknownToolError` before anything runs (see
[Section 9](#section-9--structured-tool-calling)).

**Upload validation:**
- Extension checked against `SUPPORTED_EXTENSIONS = (".csv", ".xlsx")`
  before any parsing is attempted.
- Size checked (`MAX_UPLOAD_BYTES = 10 MB`) after read, before parsing.
- The uploaded file is written to a **generated** temporary path
  (`tempfile.NamedTemporaryFile`) — the client-supplied filename is never
  used to construct a filesystem path.
- The temp file is deleted in a `finally` block, so it is removed whether
  parsing succeeded or raised.
- The client's filename is preserved only for **display**, reduced to a
  safe basename (`Path(filename).name[:120]`) with any directory
  components stripped — a filename like `"../../etc/passwd"` becomes
  `"passwd"` for display purposes and is never used as a path anywhere.

**Bounded agent steps:** `DEFAULT_MAX_STEPS = 5` — every question is
mathematically guaranteed to terminate within a fixed number of LLM calls
(see [Section 10](#section-10--agent-loop)).

**Bounded conversation history:** `DEFAULT_HISTORY_TURNS = 3`, with older
turns stripped of their row data (see
[Section 11](#section-11--conversation-state)) — a conversation cannot make
the prompt grow without bound.

**Provider error sanitization:** every exception from a provider SDK is
reduced to `LLMProviderError(f"...failed ({status_code} {type(exc).
__name__}).")` before it can reach a caller — the original exception's
message/body (which could contain request headers, account details, or
other provider-side detail) is chained via `from exc` for local debugging
only, and is never included in the message string that becomes an API
response. Verified by tests that inject an exception carrying a fake secret
string and assert it never appears in the resulting `LLMProviderError`'s
message.

**Secrets and `.env`:**
- `LLMConfig.api_key` is a dataclass field with `repr=False` — it cannot
  leak into a default `repr()`/log line via that object.
- `.env` is listed in both `.gitignore` and `.dockerignore`.
- Neither `Dockerfile.api` nor `Dockerfile.ui` contains a `COPY .env` (or
  any reference to `.env` at all beyond a documentation comment showing
  `docker run --env-file .env` for standalone use) — verified by reading
  both Dockerfiles directly (Section 16 above quotes their full content).
- Docker Compose loads `.env` only via `env_file:` at **container start**,
  never at image build time — so a built image, inspected independently of
  any running container, contains no `.env` file and no baked-in key.
- Render's environment variables are entered through the dashboard
  (`sync: false` in `render.yaml`), never committed to the repository.

**No traceback exposure:** the catch-all `Exception` handler in `main.py`
(`_unexpected`) returns a fixed, generic message
("The analyst failed to handle that request.") with HTTP `500` — the real
exception is logged server-side (`logger.exception(...)`) but never
serialized into the response.

**Result row limits:** capped at multiple layers — per-operation `limit`
arguments, `MAX_EVIDENCE_ROWS = 50` before evidence reaches the LLM, and
`MAX_QUESTION_LENGTH = 1000` characters on the inbound question itself.

**CORS:** restricted to an explicit origin list (default: only the local
Streamlit dev URLs), configurable via `CORS_ORIGINS`, not a wildcard.

**Practical threat model — what this project defends against, and what it
does not:**

*Defended against:*
- A malicious or confused model trying to run arbitrary code, an arbitrary
  SQL/pandas query string, or read files outside the sandboxed temp-file
  flow.
- A malicious upload filename used for path traversal.
- A provider outage or malformed provider response crashing the request
  handling or leaking internal detail to the client.
- An unbounded agent loop or unbounded conversation history driving up
  cost or hanging a request indefinitely.

*Not defended against (deliberately out of scope for this project — see
[Section 23](#section-23--known-limitations)):*
- **No authentication.** Any client that can reach the API can upload
  datasets and ask questions; there is no login, no API key required of
  the *caller*, and no concept of a user account.
- **No per-user isolation.** A `session_id`/`dataset_id` is an unguessable
  `uuid4().hex`, but knowledge of one is the only "access control" — there
  is no ownership check.
- **No rate limiting on the API's own inbound traffic** (only the
  *outbound* provider rate limit is handled, as an error case — nothing
  stops a client from hammering the API itself).
- **No resource quota per dataset/session** — the in-memory `Store` will
  accept datasets and sessions indefinitely until the process runs out of
  memory; there is no eviction policy (see
  [Section 23](#section-23--known-limitations)).

---

## Section 21 — Error Handling

| Error case | Where it occurs | How it's handled | What the user sees | Logs |
|---|---|---|---|---|
| Invalid file (corrupt/unreadable) | `loader.py::_read` | `DatasetReadError` raised, caught by `_dataset` handler | `400`, `{"code":"invalid_dataset", "detail": "Could not read ..."}` | None specific (not a `500`, so no `logger.exception`) |
| Unsupported extension | `validation.py::validate_source` (also re-checked in `main.py::_checked_suffix`) | `UnsupportedFileTypeError`/`DatasetError` → `_dataset` handler | `400`, message names the extension found and the supported list | None |
| Empty dataset (zero-byte or header-only) | `validation.py` | `EmptyDatasetError` → `_dataset` handler | `400` | None |
| Missing column | `validation.py::require_columns`, called from every analysis op | `ColumnNotFoundError` → `_analysis` handler (`AnalysisError` base) | `400`, lists the missing column(s) and what's available | None |
| Invalid analysis arguments (bad operator, wrong aggregation, non-numeric for a numeric op, bad `limit`) | `analysis.py` various | `InvalidOperationError` → `_analysis` handler | `400` | None |
| Malformed LLM response (unparseable JSON, missing action) | `tools.py::ToolRequest.from_text`, called from `agent.py::_next_action` | `LLMResponseError` caught **inside the agent loop** — recorded as a failed `EvidenceStep`, loop continues | Not visible as an HTTP error at all if the model recovers within the step budget; if it never recovers, the question ends as `kind="incomplete"` | None (this is expected, self-correcting behavior, not logged as an error) |
| Unknown tool | `tools.py::get_tool` | `UnknownToolError` → caught in `agent.py::_run_step`, recorded as a failed `EvidenceStep` (same self-correction path as above) | Same as above — surfaces in `steps[].error` on the eventual `AnswerResponse`, not as an HTTP error | None |
| Provider failure (network, 5xx, auth) | `gemini.py`/`groq.py::generate` | `LLMProviderError` (no `status_code`, or a non-429 one) → `_provider` handler | `502`, `{"code":"provider_error", "detail": "...failed (TypeName)."}` | `logger.warning("Provider request failed: %s", exc)` |
| Provider rate limit | Same, with `status_code == 429` detected from the SDK exception | `_provider` handler's 429 branch | `429`, `{"code":"rate_limited", "detail": "The model provider is rate-limiting requests right now. Please try again shortly."}` | `logger.warning("Provider rate-limited a request: %s", exc)` |
| Missing/invalid provider config | `llm.py::load_config`/`resolve_provider` | `LLMConfigurationError` → `_configuration` handler | `503`, `{"code":"provider_unconfigured", ...}` — the specific missing variable name is **not** included in the client-facing message | None specific |
| API unavailable (server down/unreachable) | Client-side, `api_client.py::_request` | `httpx.HTTPError` caught → `ApiError` | The Streamlit UI shows `st.error("Could not reach the analyst API at ... Is the backend running?")` | N/A (client-side; nothing server-side to log) |
| Session not found | `store.py::Store.session`/`dataset`/`drop_session` | `UnknownResource` → `_unknown` handler | `404`, `{"code":"not_found", "detail": "No session with id '...'."}` | None |
| Max agent steps reached | `agent.py::answer_question`'s `for...else` | Returns `Answer(kind="incomplete", ...)` — **not an exception at all** | `200` HTTP response, with `kind: "incomplete"` and the fixed message about the step limit | None |
| Any other unhandled exception | Anywhere in a route handler | Caught by the catch-all `Exception` handler | `500`, generic `{"code":"internal_error", "detail": "The analyst failed to handle that request."}` | `logger.exception("Unhandled error on %s %s", method, path)` — full traceback in server logs only |

---

## Section 22 — Logging and Observability

**Configuration:** `src/analyst/api/main.py` calls
`logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:
%(message)s")` at module import time, under a logger named `"analyst.api"`.
This was added in commit `af529ff` — **before that commit, there was no
logging configuration in the project at all**, and unexpected server
errors were silently swallowed with no operator-visible trace.

*Why `basicConfig` is called explicitly:* `uvicorn` configures its own
named loggers (`uvicorn`, `uvicorn.error`, `uvicorn.access`) but does not
attach a handler to the Python root logger. Without this line, this
project's own `INFO`/`WARNING` log records would fall through to Python's
"handler of last resort," which silently drops anything below `WARNING` —
so the startup log line (see below) would never actually appear in
container/deploy logs despite the code looking correct. This was caught
and fixed during Docker verification, not merely theorized.

**Startup logging:** a FastAPI `lifespan` context manager (`main.py`) logs
exactly one line when the app starts:
- If a provider is configured: `INFO:analyst.api:Starting with
  provider=<name> model=<model>` — **provider name and model name only,
  never the key.**
- If not configured: `WARNING:analyst.api:Starting with no LLM provider
  configured (set GOOGLE_API_KEY or GROQ_API_KEY).`

**Provider/model logging:** the `provider_info` callable that produces
these values (`_active_provider`, or an injected fake in tests) reads
*only* `resolve_provider()`/`load_config(...).model` — it never touches
`.api_key`, so there is no code path by which a key could end up in this
log line even by accident.

**Unexpected errors:** the catch-all exception handler calls
`logger.exception("Unhandled error on %s %s", request.method, request.url.
path)` — `logger.exception` automatically includes the full traceback in
the log record. This is the only place a full traceback exists anywhere in
the system, and it exists only in server-side logs, never in an HTTP
response (see [Section 21](#section-21--error-handling)).

**Rate-limit / provider-failure warnings:** logged at `WARNING` level with
the already-sanitized `LLMProviderError` message (type name + status code
only — see [Section 20](#section-20--security-model)) — safe to log because
that message has already had provider-specific detail stripped out before
the exception was even constructed.

**What is intentionally NOT logged:**
- The `.api_key` value, anywhere, ever — `LLMConfig.api_key` has
  `repr=False` specifically to make this hard to do by accident.
- Uploaded dataset contents/rows.
- Question text or LLM prompts/responses (no code path logs these — the
  only place a *string* representation of an answer appears in logs is
  indirectly, if an unhandled exception's traceback happens to include
  local variables containing it, which is standard Python traceback
  behavior, not something this project logs deliberately).
- Session/dataset IDs are not specifically redacted from logs, but they are
  random UUIDs, not personally identifying or secret material.

**No log aggregation, metrics, or tracing system is integrated.** Whatever
the hosting platform captures from stdout/stderr (Render's own log
viewer, or `docker compose logs`/`docker logs` locally) is the entire
observability surface. This is a portfolio-scale project, not an
instrumented production service — see
[Section 23](#section-23--known-limitations).

---

## Section 23 — Known Limitations

These are documented, deliberate scope boundaries of the current
implementation — not bugs, and not secretly planned "TODO"s hidden in the
code (nothing in the source contains a `TODO`/`FIXME` marker referencing
these).

- **In-memory datasets and sessions.** `Store` (`store.py`) is two plain
  dicts in the API process's memory. There is no database, no disk
  persistence, and no serialization anywhere in this path.
- **Sessions (and datasets) are lost on any API restart** — a deploy, a
  crash, or Render's free-tier services sleeping after inactivity (which is
  a full process stop/start, not a pause). A UI pointed at a now-dead
  `session_id` gets a `404` on its next request; there is no reconnection
  or session-recovery logic.
- **No authentication.** Anyone who can reach the deployed API can upload
  datasets and open sessions. There is no login, no API key required of
  the caller, and no per-user data isolation beyond the unguessability of
  a `uuid4()` id.
- **No persistent database, by design.** `Store`'s own docstring states
  this is deliberate for the current phase — the class is structured so a
  persistent implementation could replace it later without touching the
  route handlers, but that replacement has not been built.
- **Render free-tier services sleep when inactive**, and the first request
  after waking takes noticeably longer (the container has to start cold).
  This is a platform characteristic, not something the application
  mitigates.
- **Upload size capped at 10 MB** (`MAX_UPLOAD_BYTES`), enforced only after
  the full upload is read into memory (via Starlette's `UploadFile`, which
  spools to disk past its own internal in-memory threshold — the
  application-level check is on top of that, not a replacement for it).
- **The agent is bounded to 5 tool calls per question** — a question
  needing more steps than that terminates with `kind="incomplete"` rather
  than continuing.
- **Grounding is advisory, not enforcing** — an ungrounded answer is
  flagged, not blocked or rewritten (see
  [Section 12](#section-12--evidence-and-grounding)).
- **Conversation history is capped at 3 turns**, and only the newest of
  those keeps full result rows — a question referring to something more
  than 3 turns back will not have that context available to the model.
- **No provider fallback.** If the configured provider is down or
  rate-limited, the request fails (with a clear 429/502) rather than
  automatically retrying against the other provider. (Explicitly out of
  scope per this project's own constraints — see
  [Section 33](#section-33--future-improvement-ideas-not-implemented).)
- **Charts are a library, not an agent capability** — the LLM cannot
  request a chart; only the UI's narrow client-side heuristic
  (`_maybe_chart`) ever draws one automatically (see
  [Section 7](#section-7--visualization-engine)).
- **Docker images share one dependency list** across both services, so the
  UI image is larger than strictly necessary (carries FastAPI, Matplotlib,
  and both provider SDKs it never imports).
- **No resource quota or eviction policy** on the in-memory `Store` — every
  uploaded dataset and every created session stays in memory until the
  process restarts; a sustained high-traffic period could, in principle,
  exhaust the container's memory. Not observed in practice at the traffic
  levels this project has actually seen, but not mitigated either.

---

## Section 24 — Common Troubleshooting

| # | Problem | Likely cause | Where to check | Fix |
|---|---|---|---|---|
| 1 | UI says "Could not reach the analyst API at 127.0.0.1:8000" | `ANALYST_API_URL` is unset or wrong, or the API isn't running | The sidebar's "API URL" field in the UI (it's editable at runtime); the `ANALYST_API_URL` env var the UI process actually has | Set `ANALYST_API_URL` to the API's real address, or edit the field directly in the UI and it takes effect on the next request. |
| 2 | `/api/health` doesn't respond | API process not running, wrong port, or a firewall/network issue | `docker compose ps` / Render's service logs and status page; try `curl <api-url>/api/health` directly | Restart/redeploy the API service; check container logs for a startup exception. |
| 3 | Render API deployment fails | A Docker build error, or a missing/invalid environment variable causing a startup crash | Render's build logs and deploy logs for the `ai-data-analyst-api` service | Reproduce the build locally: `docker build -f Dockerfile.api -t test .`; check that the required env vars are actually set in the Render dashboard. |
| 4 | Render UI deployment fails | Same class of issue as #3, on `Dockerfile.ui` | Render's build/deploy logs for `ai-data-analyst-ui` | `docker build -f Dockerfile.ui -t test .` locally to reproduce. |
| 5 | Groq returns 429 | Provider-side rate limit hit | The API's `429` response body (`code: "rate_limited"`); server logs show `Provider rate-limited a request: ...` | Wait and retry; this is handled gracefully already (friendly message, correct status code) — not a bug to fix, just a rate limit to respect. |
| 6 | Groq model unavailable/deprecated | `GROQ_MODEL_NAME` points at a model Groq no longer serves | The API's error response (`provider_error`, mentioning a status/type from the SDK); Groq's own model list | Update `GROQ_MODEL_NAME` in the environment (local `.env`, or the Render dashboard) to a currently-supported model. |
| 7 | Gemini rate limit | Same class as #5, on the Gemini side (`status_code == 429` from `google-genai`) | Same handling path — `429`, `rate_limited` | Same as #5. |
| 8 | Dataset upload fails | Wrong extension, empty file, corrupt file, or over 10 MB | The API's `400` response `detail`, which names the specific problem (`Unsupported file type`, `The uploaded file is empty`, `larger than 10 MB`, etc.) | Fix the file/format per the message; there's no way to raise the 10 MB limit without editing `MAX_UPLOAD_BYTES` in `main.py` (an application code change). |
| 9 | LLM returns malformed response | The model didn't follow the JSON-shape instructions in `AGENT_RULES`/`ANSWER_RULES` | This is handled automatically inside the agent loop for the tool-selection step (recorded as a failed step, model gets another try) — it is *not* separately handled for the final-answer step, which has no JSON requirement to violate | Usually self-corrects within the step budget; if it doesn't, the question ends as `kind="incomplete"`. |
| 10 | Agent reaches max steps | The question genuinely needs more than 5 tool calls, or the model keeps failing/retrying without making progress | `AnswerResponse.kind == "incomplete"`; `steps` in the response shows every attempt and its error, if any | Ask a narrower question ("ask for one specific figure at a time" — the app's own suggested fix); raising `DEFAULT_MAX_STEPS` is an application code change. |
| 11 | Follow-up doesn't work | The follow-up needs context from more than 3 turns back, or the session was lost (see #16) | `Conversation.history_turns` (default 3); check the session still exists (`GET /api/sessions/{id}`) | Ask again with the needed context restated, or increase `DEFAULT_HISTORY_TURNS`/`history_turns` (a code change). |
| 12 | Docker API doesn't start | A build failure, a port already in use on the host, or a genuinely broken image | `docker compose logs api`; `docker compose ps` for the container's status | Rebuild (`docker compose build`); if it's a port conflict, use `API_HOST_PORT=<other-port> docker compose up`. |
| 13 | Docker UI can't reach API | `ANALYST_API_URL` inside the `ui` container is wrong (e.g. set to `localhost` instead of `api`) | `docker-compose.yml`'s `ui.environment.ANALYST_API_URL` (should be `http://api:8000`, never `localhost`/`127.0.0.1`) | Confirm the compose file wasn't edited to use a loopback address — see [Section 16](#section-16--docker-architecture) for why that specifically breaks. |
| 14 | `$PORT` issue on Render | Unlikely given the current Dockerfiles, but would present as the service failing its health check because it's listening on the wrong port | The Dockerfile's `CMD` line (should be shell-form with `${PORT:-...}`, not exec-form with a hardcoded port) | Confirm `CMD` is written as documented in [Section 16](#section-16--docker-architecture); an exec-form `CMD ["uvicorn", ..., "--port", "8000"]` would silently ignore `$PORT`. |
| 15 | CORS issue | A browser-based client other than the deployed UI itself is calling the API directly from an origin not in `CORS_ORIGINS` | Browser console (CORS errors are browser-side); `CORS_ORIGINS` env var on the API service | Add the calling origin to `CORS_ORIGINS` (comma-separated). Note this does **not** affect the deployed UI's own traffic, since the UI calls the API server-side, not from browser JS — see [Section 15](#section-15--api-client). |
| 16 | Session disappears | The API process restarted (deploy, crash, or free-tier sleep/wake) — see [Section 23](#section-23--known-limitations) | Requests against the old `session_id` return `404` | Start a new session (the UI's "Start a new session" button does this, or `POST /api/sessions` again). This is expected behavior, not a bug. |
| 17 | Chart doesn't appear | The result shape doesn't match the UI's narrow heuristic — anything other than exactly one label column + one numeric column, 2–50 rows | `app.py::_maybe_chart`'s conditions | This is by design (see [Section 7](#section-7--visualization-engine)) — not every result is charted; the data table is always shown regardless. |
| 18 | README/live URL becomes outdated | A redeploy changed the Render subdomain, or the service was recreated | Render's dashboard for the actual current URLs | Update `README.md` (and this document's header) with the new URLs; there is no automatic mechanism keeping documentation in sync with a live deployment's URL. |

---

## Section 25 — How to Add a New Analysis Tool

**This is a walkthrough, not an implementation.** Using a hypothetical
`top_n` tool (return the top N rows by a numeric column) as the running
example — **`top_n` does not exist in this codebase.**

1. **Analysis implementation** — add a function to `src/analyst/analysis.py`
   following the existing pattern: validate arguments (reuse
   `require_columns`/`require_numeric` from `validation.py`, raise
   `InvalidOperationError`/`ColumnNotFoundError`), do the pandas work,
   return an `AnalysisResult` via the same `_records`/metadata pattern the
   other seven functions use. `top_n(frame, column, n=5)` would likely be a
   thin wrapper that's really just `sort_rows` with a fixed `ascending=
   False, limit=n)` — worth checking whether a genuinely new function is
   needed at all, or whether the existing `sort_rows` already covers the
   use case (this specific example probably doesn't justify a new
   function — a more genuinely new operation, e.g. a rolling-window
   calculation, would).
2. **Argument validation** — happens twice, intentionally: once inside the
   new `analysis.py` function itself (so the function is safe to call
   directly, outside the LLM path too), and once declaratively in the
   `ToolArgument` list described in step 4 (so the dispatcher can reject a
   bad call *before* invoking the function).
3. **Result conversion** — reuse `conversion.to_python`/`to_optional_float`
   for anything numpy/pandas-typed, exactly as every existing operation
   does — never return a raw numpy scalar or `pd.Timestamp` in a result row.
4. **Registry registration** — add a new `ToolSpec` entry to the `TOOLS`
   dict-comprehension in `src/analyst/tools.py`: `name`, a one-line
   `purpose` string, a tuple of `ToolArgument(name, types, description,
   required, allowed_values)` for every parameter, and `function=top_n`
   (the direct function reference — not a string).
5. **LLM tool schema/prompt** — nothing to do manually here:
   `describe_tools()` automatically includes every `TOOLS` entry, and
   `AGENT_RULES`/`build_agent_prompt` in `prompts.py` already say "Request
   only tools from the supplied list" generically — a new tool is visible
   to the model automatically once it's in `TOOLS`.
6. **Dispatcher behavior** — also automatic: `run_tool`/`validate_
   arguments` work generically off `ToolSpec`/`ToolArgument`, so a new
   entry needs no changes to `tools.py`'s dispatch logic itself.
7. **Tests** — add a new `tests/test_analysis.py` block for the function
   in isolation (unit tests, no LLM involved — this is the majority of
   existing coverage style), plus at least one `tests/test_tools.py` case
   dispatching it through `run_tool`, plus (if the new tool changes
   end-to-end agent behavior in a way worth locking down) a
   `tests/test_agent.py` case using `FakeLLM` to script a call to it.
8. **Agent compatibility** — nothing to change in `agent.py` itself; it is
   already fully generic over whatever `describe_tools()`/`run_tool`
   expose.
9. **Documentation** — add the tool name to the analysis-tools list in
   `README.md` and to [Section 6](#section-6--analysis-engine) of this
   document.
10. **UI implications** — usually none required. `app.py::_render_answer`
    already renders any result's `rows` as a generic table, and
    `_maybe_chart` already applies its shape-based heuristic to whatever
    comes back — a new tool that returns the same "rows of dicts" shape
    other tools do needs no UI changes. A tool whose natural output isn't
    tabular (unlikely, given `AnalysisResult`'s fixed shape) would be the
    exception.

---

## Section 26 — How to Add a New LLM Provider

Using a hypothetical `"anthropic"` provider as the example — **not
implemented.**

1. **`LLMClient` interface** — the new provider's client class must
   implement exactly one method: `generate(self, prompt: str, *, system:
   str | None = None, json_output: bool = False) -> str`. Nothing else is
   required by any caller in this codebase (verified: `LLMClient` in
   `llm.py` is the entire `Protocol`, and both `agent.py` and `api/main.py`
   depend only on this shape).
2. **Provider implementation** — a new file, `src/analyst/anthropic.py` (by
   the existing naming convention — one file per provider), following
   `gemini.py`/`groq.py`'s pattern exactly: a class taking an `LLMConfig`
   in its constructor, lazily importing the SDK inside `__init__` (so the
   SDK dependency is optional unless this provider is actually used),
   wrapping every SDK exception into `LLMProviderError` (reading a numeric
   status code off the SDK's exception if one exists, exactly as `gemini.
   py`/`groq.py::_numeric_status` do), and raising `LLMResponseError` on an
   empty reply.
3. **Provider settings** — add one entry to `PROVIDER_SETTINGS` in
   `llm.py`: `ProviderSettings("anthropic", "ANTHROPIC_API_KEY",
   "ANTHROPIC_MODEL_NAME", "<some-default-model>")`. Add a corresponding
   `ANTHROPIC = "anthropic"` constant and include it in `PROVIDERS`.
4. **Configuration** — nothing else to change in `load_config`/
   `resolve_provider` — both are already fully generic over
   `PROVIDER_SETTINGS`.
5. **Environment variables** — document `ANTHROPIC_API_KEY`,
   `ANTHROPIC_MODEL_NAME` in `.env.example`, `README.md`, and
   [Section 18](#section-18--environment-configuration) of this document.
6. **`create_client`** — add one `elif`/branch in `llm.py::create_client`
   mapping the provider name to a lazy import of the new class, matching
   the existing `if chosen == GEMINI: ... else: from analyst.groq import
   GroqClient` pattern (which would need to become an explicit `elif`
   chain, or a small dict of provider-name → import-function, once there
   are three providers).
7. **Tests** — a new `tests/test_anthropic.py` following `test_gemini.py`'s
   structure exactly: stub the SDK import point via `monkeypatch`, test
   the reply path, the system-prompt path, the JSON-output flag, provider
   failures (with and without a status code), an empty reply, and a bad
   client-construction failure. Add the new provider to the relevant
   cross-provider tests in `test_providers.py` (provider resolution,
   `create_client` dispatch).

No changes to `agent.py`, `conversation.py`, `tools.py`, or the API layer
are needed — this is the entire point of the `LLMClient` abstraction (see
[Section 8](#section-8--llm-architecture) and
[Section 34](#section-34--interview--technical-discussion-notes)).

---

## Section 27 — How to Modify the Agent

**Where things live:**

| To change... | Edit |
|---|---|
| The rules the model is told (what "answer"/"clarification"/"call_tool" mean, tone, constraints) | `AGENT_RULES`, `ANSWER_RULES`, `CLARIFICATION_RULES` in `prompts.py` |
| How the "next action" prompt is assembled (what context/history/steps are shown) | `build_agent_prompt` in `prompts.py` |
| How the "final answer" prompt is assembled | `build_answer_prompt` in `prompts.py` |
| The step budget | `DEFAULT_MAX_STEPS` in `agent.py` (or pass `max_steps=` explicitly at the `answer_question`/`Conversation` call site) |
| The clarification/answer/call_tool action shapes themselves | `ToolRequest`/`ACTIONS`/`MESSAGE_ACTIONS` in `tools.py` |
| The loop's control flow (when it breaks, returns early, records a failure) | `answer_question` in `agent.py` |
| How much conversation history is kept, and how it's summarized | `DEFAULT_HISTORY_TURNS`, `_SUMMARY_FIELDS`, `Conversation.history()` in `conversation.py` |
| How evidence rows are capped before reaching the model | `MAX_EVIDENCE_ROWS`, `trim_result` in `prompts.py` |
| The grounding check's sensitivity (rounding tolerance, ignored numbers) | `_IGNORED`, `_supported` in `grounding.py` |

**What to be careful not to break:**

- **Provider abstraction** — the agent must keep depending only on
  `LLMClient.generate(prompt, system=, json_output=)`. Do not add
  provider-specific branching inside `agent.py`/`conversation.py`; that
  belongs in `gemini.py`/`groq.py` only.
- **Safe tool dispatch** — never add a path that calls a Python function by
  a name string outside `tools.TOOLS` (e.g. `getattr(analysis_module,
  tool_name)`), and never add `eval`/`exec` as a shortcut for "let the
  model do more." Any new capability the model needs should be a new,
  explicitly-registered tool (see
  [Section 25](#section-25--how-to-add-a-new-analysis-tool)), not a
  loosening of the dispatcher.
- **Bounded execution** — if the step budget or the loop structure changes,
  keep the `for` loop's fixed iteration count as the termination
  guarantee. Don't introduce a `while True` or a retry mechanism that
  could re-enter the loop with a fresh budget for the same question.
- **Grounding** — if the final-answer prompt changes, keep
  `check_grounding` being called on the actual final text against the
  actual evidence used — don't let the two drift (e.g. don't add a code
  path where the model can see evidence that never gets passed into the
  grounding check).
- **Conversation history bounding** — if `Conversation.history()` changes,
  keep both bounds (turn count *and* the "only the newest turn keeps
  rows" rule) — removing either reintroduces the risk of the prompt
  growing without bound across a long conversation.

---

## Section 28 — How to Redeploy From Scratch

Assumes: the GitHub repository, a Render account, and a provider API key
(Gemini or Groq) — but no memory of the current deployment's specific
service names or URLs.

1. **Clone / have the repository available:**
   ```bash
   git clone https://github.com/KD-kaustubh/AI-DATA-ANALYST-AGENT.git
   cd AI-DATA-ANALYST-AGENT
   ```
2. **Confirm the provider key you'll use** — have either a Google AI
   Studio (Gemini) key or a Groq console key ready. Do not paste it
   anywhere in the repository or a commit.
3. **On Render:** "New +" → "Blueprint" → connect/select this GitHub
   repository. Render detects `render.yaml` at the repo root and proposes
   two services (`ai-data-analyst-api`, `ai-data-analyst-ui`).
4. **Configure the API service's environment variables** (Render
   dashboard → the `ai-data-analyst-api` service → Environment):
   - Set `LLM_PROVIDER` to `groq` or `gemini`.
   - Set the matching key (`GROQ_API_KEY` or `GOOGLE_API_KEY`).
   - Optionally set the matching `*_MODEL_NAME` variable if you want a
     model other than the default (`llama-3.3-70b-versatile` for Groq,
     `gemini-2.5-flash` for Gemini).
5. **Deploy the API service** and wait for the build to finish.
6. **Verify the API:**
   ```bash
   curl https://<your-api-service>.onrender.com/api/health
   ```
   Expect `{"status":"ok","version":"0.1.0","provider":"<your provider>",
   "model":"<your model>"}`.
7. **Configure the UI service's environment variable**
   (`ai-data-analyst-ui` → Environment): set `ANALYST_API_URL` to the exact
   URL from step 6 (no trailing slash).
8. **Deploy the UI service.**
9. **Verify the UI:** open `https://<your-ui-service>.onrender.com` in a
   browser; confirm the "🤖 AI Model: ..." caption near the top shows the
   provider/model you configured (this is read live from `/api/health`,
   proving the two services are correctly connected).
10. **Test the application end to end:** upload a small CSV, confirm the
    profile renders, ask a simple question ("What is the average of
    <numeric column>?"), confirm a grounded answer comes back, ask a
    follow-up, confirm it uses the prior context.

If any step fails, see [Section 24](#section-24--common-troubleshooting)
for the specific symptom.

---

## Section 29 — Local Development Workflow

```bash
# 1. Clone
git clone https://github.com/KD-kaustubh/AI-DATA-ANALYST-AGENT.git
cd AI-DATA-ANALYST-AGENT

# 2. Virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

# 3. Install (this installs the project itself, editable, plus dev extras)
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# edit .env: set LLM_PROVIDER and one of GOOGLE_API_KEY / GROQ_API_KEY

# 5. Run the tests
pytest

# 6. Run the backend (terminal 1)
uvicorn analyst.api:app --reload    # http://127.0.0.1:8000, docs at /docs

# 7. Run the frontend (terminal 2, same venv activated)
streamlit run app.py                # http://localhost:8501
```

**Which process runs where:** the API (`uvicorn`) and the UI (`streamlit`)
are two independent, long-running processes, each needing the virtualenv
activated in its own terminal — `uvicorn analyst.api:app --reload` will
auto-reload on source changes to `src/analyst/`; Streamlit auto-reloads
`app.py`/`api_client.py` changes on its own by default.

**Docker alternative** (no local Python environment needed at all beyond
Docker itself):
```bash
cp .env.example .env   # then fill in a key
docker compose up --build
```
See [Section 16](#section-16--docker-architecture) for full detail,
including the `API_HOST_PORT` override if port 8000 is already taken
locally.

**Why `pip install -r requirements.txt` and not just `pip install -e .`:**
`requirements.txt` is `-e .[dev]` — installing the `dev` extra is what
pulls in `pytest` and `httpx2` (needed for `fastapi.testclient.TestClient`
under the Starlette version this project pins against). `pip install -e .`
alone would install the project without test dependencies.

---

## Section 30 — Git / Development Workflow

Reconstructed directly from the actual commit history (`git log`), which
is a clean, linear, single-branch history — 14 commits, all on `main`, no
merge commits, no force-pushes, no rewritten history (confirmed: no
`filter-branch`/`rebase` artifacts, and every commit's parent is exactly
the prior commit).

**Observed conventions:**
- **Everything on `main`.** No feature branches were used or are present
  in the repository.
- **Normal pushes only.** No evidence of force pushes (a force push would
  typically leave gaps or an inconsistent history; this one is fully
  linear).
- **Logical, phase-sized commits**, not one-commit-per-file or
  one-commit-per-line-change — each commit corresponds to a complete,
  working unit of functionality (see [Section 2](#section-2--project-history--development-roadmap)
  for the mapping). Small later fixes (`ff84055`, `8c40327`, `2d6c562`)
  are still each a single, complete fix, not split further.
- **Commit message style:** short, imperative, natural-sounding
  ("Add dataset loading and profiling", "Improve rate limit error
  handling") — no robotic/enterprise phrasing, no ticket numbers, no
  emoji, no AI/session trailers of any kind (no `Co-Authored-By`, no
  generated-by footer) in any of the 14 commits.
- **Tests before commit.** Every phase's commit corresponds to a state
  where the full test suite passes (`pytest` run and confirmed
  342/342 as of this document, and the project's own history shows the
  passing count growing phase over phase as functionality was added —
  not independently re-verified per historical commit, since checking
  out and testing every past commit was outside the scope of producing
  this document).
- **`git diff --check`** was used as a pre-commit whitespace/conflict-marker
  check in this project's own working process (evidenced by the pattern of
  changes across phases; not itself stored in git history, since it's a
  verification step rather than a file).
- **Secret checks before every push** — grepping the diff for
  key-shaped strings (`AIza...`, `gsk_...`) before committing; this is a
  process discipline, not a repository artifact, but is the reason no
  secret has ever appeared in this repository's history.
- **Docker verification when relevant** — commits touching
  `Dockerfile.*`/`docker-compose.yml`/`render.yaml` were paired with an
  actual `docker compose build`/`up`/health-check verification before being
  committed, not merely written and assumed correct.

**If continuing this project's conventions:** keep commits phase-sized and
working at each commit, keep messages short and natural, keep everything
on `main` unless a change is large/risky enough to warrant a branch (not
done so far in this project's history), run the full test suite and
`git diff --check` before committing, and never commit `.env` or any file
containing a real key.

---

## Section 31 — Verified Production Checklist

Only items actually confirmed (by direct test run, direct HTTP check, or
direct source inspection while producing this document) are checked.

- [x] GitHub repository exists and is reachable
      (`https://github.com/KD-kaustubh/AI-DATA-ANALYST-AGENT`, verified
      `200`)
- [x] README present and describes the deployed state (`README.md`,
      last updated commit `db6886c`)
- [x] MIT License present (`LICENSE`, root of repo)
- [x] 342 tests passing (`pytest`, re-run while producing this document)
- [x] Docker images build successfully (`Dockerfile.api`, `Dockerfile.ui` —
      verified during Phase 6.1 and again during the production-readiness
      pass; not re-built again specifically for this document)
- [x] API deployed (`https://ai-data-analyst-api-8b3n.onrender.com`,
      verified `200` on `/api/health`)
- [x] UI deployed (`https://ai-data-analyst-ui-tfvu.onrender.com`,
      verified `200`)
- [x] API health check returns the expected shape and values (`{"status":
      "ok","version":"0.1.0","provider":"groq","model":"openai/gpt-oss-
      120b"}`, verified directly)
- [x] Groq is the active production provider/model (confirmed via the
      health check above)
- [x] Dataset upload works (verified against the live API earlier in this
      project's work — not re-uploaded again specifically for this
      document, since the task instructions for this document forbid
      unnecessary repeated actions against the live app; the capability is
      also covered by 47 passing `test_api.py` tests against the same code
      path)
- [x] Real LLM question works (verified against live Groq earlier in this
      project's work — see [Section 2](#section-2--project-history--development-roadmap))
- [x] Follow-up question works (same verification — a genuine two-turn
      conversation was run against the live deployment)
- [x] Grounding works (`is_grounded: true` observed on a real verified
      answer; the unsupported-figure path is covered by
      `test_conversation.py`/`test_agent.py`)
- [x] Security checks pass (no `eval`/`exec`/`subprocess`/etc. in source;
      no secret-shaped string in any tracked file — both verified by
      direct search while producing this document)
- [x] Render deployment configuration present and matches the live
      services (`render.yaml`, cross-checked against the live URLs above)
- [x] Git working tree clean before this document's own commit (`git
      status -sb` showed only the pre-existing, unrelated `.gitignore`
      change — see this document's own commit for the state after it was
      added)

---

## Section 32 — Important File Reference

| File | Purpose | When to modify |
|---|---|---|
| `src/analyst/loader.py` | File → DataFrame | Adding a new supported file format. |
| `src/analyst/validation.py` | File/frame/column checks shared across the codebase | Changing what counts as a valid dataset, or adding a new shared validation rule. |
| `src/analyst/profiling.py` | Builds `DatasetProfile` | Changing what's detected/reported about a dataset (dtypes, dates, warnings). |
| `src/analyst/analysis.py` | The 7 deterministic pandas operations | Adding/changing an analysis operation — see [Section 25](#section-25--how-to-add-a-new-analysis-tool). |
| `src/analyst/charts.py` | The 6 chart builders | Adding/changing a chart type (library-level; not currently agent-callable). |
| `src/analyst/tools.py` | Tool registry, `ToolRequest`, dispatcher | Registering a new tool, changing argument validation, changing the `call_tool`/`answer`/`clarification` action shapes. |
| `src/analyst/prompts.py` | The three prompts (agent, answer, clarification rules) | Tuning model behavior/instructions. |
| `src/analyst/agent.py` | The bounded multi-step loop | Changing the loop's control flow, step budget, or evidence handling — see [Section 27](#section-27--how-to-modify-the-agent). |
| `src/analyst/conversation.py` | Multi-turn session state | Changing history bounding or turn behavior. |
| `src/analyst/grounding.py` | Numeric grounding check | Changing what counts as "supported," rounding tolerance, or ignored numbers. |
| `src/analyst/llm.py` | Provider-agnostic config/resolution | Adding a new provider's settings entry — see [Section 26](#section-26--how-to-add-a-new-llm-provider). |
| `src/analyst/gemini.py`, `groq.py` | Provider-specific clients | Fixing/updating one provider's SDK integration without touching the other. |
| `src/analyst/api/main.py` | FastAPI app, routes, error handling, CORS, logging | Adding/changing an endpoint, changing error-to-HTTP-status mapping, changing logging. |
| `src/analyst/api/schemas.py` | Pydantic request/response shapes | Changing what an API response includes. |
| `src/analyst/api/store.py` | In-memory dataset/session storage | Changing persistence (e.g. adding eviction, or swapping in a real database — this is the intended seam for that). |
| `app.py` | Streamlit UI | Any UI/UX change. |
| `api_client.py` | The UI's HTTP client | Changing how the UI talks to the API (timeouts, error formatting, base URL resolution). |
| `Dockerfile.api`, `Dockerfile.ui` | Container images | Changing runtime dependencies, base image, or how `$PORT`/CMD works. |
| `docker-compose.yml` | Local two-service stack | Changing local dev networking/ports/healthcheck. |
| `render.yaml` | Render Blueprint | Changing which env vars are declared for deployment, health check path, or plan. |
| `pyproject.toml` | Dependencies, build config, pytest config | Adding/removing a dependency, changing pytest's warning filters. |
| `.env.example` | Documented environment variables (placeholders only) | Whenever a new env var is introduced anywhere in the project. |
| `tests/conftest.py` | Shared fixtures (`sample_frame`, `analysis_frame`, `FakeLLM`, `BrokenLLM`) | Adding a new commonly-needed test fixture. |
| `README.md` | Portfolio-facing documentation | Whenever the live deployment, features, or public-facing setup instructions change. |
| `docs/PROJECT_HANDOFF.md` | This document | Whenever the architecture, deployment, or any verified fact in this document changes. |

---

## Section 33 — Future Improvement Ideas (NOT IMPLEMENTED)

The following are logical extensions of the current architecture. **None
of them exist in the codebase today.** They are listed here as context for
future planning, not as a roadmap requiring near-term work.

- **Persistent storage** for datasets/sessions (e.g. swapping `Store`'s
  in-memory dicts for a real database) — the class is already structured
  as a seam for this, per its own docstring, but no persistent backend has
  been built.
- **Authentication** — no login or per-caller identity exists today; adding
  one would also require deciding on session/dataset ownership semantics
  that don't currently exist.
- **Better multi-user isolation** — related to the above; today isolation
  is only "you need to know the UUID."
- **Larger dataset processing** — the current design loads a whole file
  into memory (both at upload time and for every subsequent analysis
  operation); genuinely large files would need streaming/chunked
  processing, which is not implemented.
- **Background jobs** — every request today is handled synchronously
  within the HTTP request/response cycle (including the LLM round trip);
  a long-running analysis has no async job/polling pattern.
- **Richer visualizations** — `analyst.charts` already has six chart types,
  but none are agent-callable; making charts a registered tool (so the
  model could request one) would be a natural, contained extension
  following the same pattern as [Section 25](#section-25--how-to-add-a-new-analysis-tool).
- **Additional analysis tools** — e.g. pivot tables, rolling
  windows/time-series smoothing, outlier detection — following the same
  registration pattern as the existing seven.
- **Provider fallback** — automatically retrying against a second provider
  if the primary one is down/rate-limited. **Explicitly out of scope per
  this project's stated constraints during development** — not merely
  unimplemented by oversight.
- **Caching** — e.g. caching a `DatasetProfile` beyond the current
  per-process, per-`Store`-entry caching (`Store.add_dataset` already
  profiles once and reuses it — but there is no cross-request caching of
  analysis *results* for repeated identical questions).

---

## Section 34 — Interview / Technical Discussion Notes

Grounded in the actual implementation — no invented benchmark numbers, no
claims beyond what the code and tests demonstrate.

**Why deterministic tools instead of arbitrary LLM-generated Python?**
Because an LLM is a fluent text generator, not a calculator — it will
produce a plausible-looking wrong number as confidently as a right one. By
restricting the model to *selecting* from seven fixed, pre-validated pandas
functions with typed arguments, every number in the system's output is
traceable to a real pandas computation, and the attack surface for
arbitrary code execution is eliminated entirely (verified: zero `eval`/
`exec`/`subprocess` anywhere in the codebase).

**Why structured tool calling (JSON actions) rather than free-form
function-calling APIs some providers offer natively?** Using a
provider-agnostic JSON contract (`{"action": "call_tool", ...}`) parsed by
this project's own code, rather than each provider's native tool-calling
feature, keeps the `LLMClient` interface to a single `generate(prompt) ->
str` method — meaning any provider that can follow a system prompt and
return text works, without needing to separately implement each provider's
different native function-calling schema/API.

**Why provider abstraction (`LLMClient`)?** So the rest of the system
(agent, conversation, API) depends on one interface, not a specific SDK.
Concretely demonstrated by this project's own history: Groq support
(Phase 3.1) was added without touching `agent.py`, `tools.py`, or the API
layer at all — only `llm.py` (one new `ProviderSettings` entry) and one new
file (`groq.py`) were needed.

**Why bounded agent steps?** Two reasons, both real constraints: cost (each
step is a paid LLM call) and reliability (an unbounded loop risks never
terminating if the model keeps making the same mistake). `DEFAULT_MAX_STEPS
= 5` is a deliberate trade-off — enough for a genuinely multi-part question
plus one self-correction, capped low enough to bound worst-case latency and
cost per question.

**Why evidence tracking (`EvidenceStep`)?** It's what makes the final
answer auditable — every tool call this question made, successful or
failed, with its exact arguments and result, is retained and exposed
(`AnswerResponse.steps` in the API). This is also what lets a failed step
become a *correction opportunity* rather than a dead end: the model sees
its own prior mistake on the next iteration.

**Why grounding?** Tool calls guarantee every *individual number* is
correct at the point it's computed — but the final answer is still
free-form generated text, which is a re-transcription step where an LLM
could still (in principle) drop a digit, swap a value, or invent something
while paraphrasing. Grounding is a lightweight, honestly-scoped safety net
specifically on that one remaining risk — explicitly not a proof of
correctness (its own docstring says so), just a check that flags anything
suspicious rather than silently trusting the wording.

**Why FastAPI + Streamlit specifically?** FastAPI gives async-capable,
typed (Pydantic), self-documenting (automatic OpenAPI/`/docs`) HTTP
endpoints with minimal boilerplate; Streamlit gives a usable chat-style UI
without writing any frontend JavaScript, at the cost of the UI being a
Python server-rendered app rather than a client-side SPA. The two
communicate over plain HTTP — the UI has no special access to the backend's
internals, which keeps the architecture honest about what a "real" client
of this API would also be able to do.

**Why in-memory sessions?** Simplicity, and honesty about scope — this is
explicitly a single-instance, demo/portfolio-scale deployment, not a
multi-tenant SaaS product. Adding a database before there's a genuine need
for persistence (multi-instance deployment, session durability across
restarts) would be premature complexity for what this project actually is
today; `Store` is deliberately structured so that a persistent
implementation could be dropped in later without touching the route
handlers.

**How is file upload secured?** Extension allow-list, a size cap, and —
critically — the uploaded file is never written to a path derived from the
client's filename; a generated temporary path is used and deleted
immediately after parsing, success or failure. The client's filename is
reduced to a safe display-only basename.

**How are provider failures handled?** Every SDK exception is caught at
the provider boundary (`gemini.py`/`groq.py`) and normalized into
`LLMProviderError`, carrying only the exception's type name and (if the SDK
exposed one) a numeric status code — never the original message, which
could carry request/account detail. A `429` specifically is detected and
mapped to a distinct, friendlier HTTP response.

**How does the agent correct a failed tool call?** A failure — unknown
tool, bad argument, hallucinated column — is caught and turned into a
recorded `EvidenceStep` with an `error` message, not raised as an
exception. That step, error message included, is shown back to the model
in the very next prompt (`build_agent_prompt` includes "Steps already
taken for this question"), so the model can read what went wrong and try a
corrected call on its next turn — still within the same fixed step budget.

**How does a follow-up question work?** `Conversation.ask()` calls
`answer_question` with `history=self.history()` — the last few turns
(default 3), with the most recent turn's full result rows included and
older turns reduced to a metadata-only summary. The model reads this
history in the same prompt as the current question and can answer directly
from it (no new tool call) when the prior evidence already suffices — as
verified in real testing, where "Which department has the highest average
salary?" was answered correctly purely from the previous turn's grouped
result, without re-running any aggregation.

**How would the architecture scale?** Honestly: the current design is
single-instance and stateful in-process (`Store` is a plain dict in one
Python process's memory), which is the main limiter — running multiple
instances behind a load balancer would immediately break session
continuity, since a session created on instance A would be invisible to
instance B. Scaling this out would require externalizing `Store`'s state
(the class is already structured as a seam for that — see
[Section 33](#section-33--future-improvement-ideas-not-implemented)) before
horizontal scaling would work correctly. No performance numbers (latency,
throughput) have been benchmarked for this project, so none are claimed
here.

---

## Section 35 — Final Project Snapshot

```
Project:      AI Data Analyst Agent
Repository:   https://github.com/KD-kaustubh/AI-DATA-ANALYST-AGENT.git
Live UI:      https://ai-data-analyst-ui-tfvu.onrender.com
API:          https://ai-data-analyst-api-8b3n.onrender.com
Health:       https://ai-data-analyst-api-8b3n.onrender.com/api/health
Provider:     Groq
Model:        openai/gpt-oss-120b
Tests:        342 passed, 0 failed (verified while producing this document)
Deployment:   Render (two independent Docker web services)
License:      MIT
Status:       Completed / deployed / verified

Last commit at the time this document was written:
  db6886cd76ca7cb0e3541408199a5e106e44aa35 — "Update README for deployed app"
  (this handoff document's own commit follows immediately after)
```
