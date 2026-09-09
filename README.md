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

Set `GOOGLE_API_KEY` in a `.env` file (copy `.env.example`), then let a model
pick the tool while pandas does the arithmetic.

```python
from analyst import GeminiClient, answer_question, load_dataset

frame = load_dataset("data/sales.csv")
answer = answer_question(frame, "Which region earns the most?", GeminiClient())

print(answer.text)       # the reply in plain language
print(answer.tool)       # which analysis tool ran
print(answer.evidence)   # the AnalysisResult it was written from
```

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
