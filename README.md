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

## Development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```
