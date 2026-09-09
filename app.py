"""Streamlit front end for the AI Data Analyst Agent.

Run with: streamlit run app.py

This module holds no analysis logic. It uploads a file, opens a session and
asks questions over HTTP; the backend does the rest.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from api_client import AnalystApi, ApiError, api_base_url

PAGE_TITLE = "AI Data Analyst"
KIND_LABELS = {
    "answer": ("✅", "Answer"),
    "clarification": ("❓", "Needs clarification"),
    "incomplete": ("⚠️", "Unfinished"),
}


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="📊", layout="wide")
    st.title("📊 AI Data Analyst")
    st.caption(
        "Ask questions about your data in plain English. "
        "Every figure is computed with pandas, not written by the model."
    )

    _init_state()
    api = AnalystApi(st.session_state.api_url)

    with st.sidebar:
        _sidebar(api)

    if not st.session_state.dataset:
        st.info("Upload a CSV or XLSX file in the sidebar to begin.")
        return

    _dataset_panel(st.session_state.dataset)
    st.divider()
    _conversation_panel(api)


def _init_state() -> None:
    defaults = {
        "api_url": api_base_url(),
        "dataset": None,
        "session_id": None,
        "history": [],
        "uploaded_name": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _sidebar(api: AnalystApi) -> None:
    st.header("Dataset")
    st.text_input("API URL", key="api_url", help="Where the backend is running.")

    upload = st.file_uploader("CSV or XLSX", type=["csv", "xlsx"])
    if upload is not None and upload.name != st.session_state.uploaded_name:
        _load_dataset(api, upload.name, upload.getvalue())

    dataset = st.session_state.dataset
    if not dataset:
        return

    st.success(f"Loaded **{dataset['filename']}**")
    st.caption(f"Dataset `{dataset['dataset_id'][:8]}…`")
    if st.session_state.session_id:
        st.caption(f"Session `{st.session_state.session_id[:8]}…`")

    if st.button("Start a new session", use_container_width=True):
        _start_session(api, dataset["dataset_id"], reset_history=True)

    st.divider()
    st.caption(
        "Sessions live in the backend's memory and disappear when it restarts."
    )


def _load_dataset(api: AnalystApi, filename: str, payload: bytes) -> None:
    with st.spinner("Validating and profiling…"):
        try:
            dataset = api.upload_dataset(filename, payload)
        except ApiError as exc:
            st.error(str(exc))
            return

    st.session_state.dataset = dataset
    st.session_state.uploaded_name = filename
    st.session_state.history = []
    _start_session(api, dataset["dataset_id"], reset_history=True)


def _start_session(api: AnalystApi, dataset_id: str, *, reset_history: bool) -> None:
    try:
        session = api.create_session(dataset_id)
    except ApiError as exc:
        st.session_state.session_id = None
        st.error(str(exc))
        return

    st.session_state.session_id = session["session_id"]
    if reset_history:
        st.session_state.history = []


def _dataset_panel(dataset: dict[str, Any]) -> None:
    st.subheader("Dataset profile")

    left, middle, right = st.columns(3)
    left.metric("Rows", f"{dataset['row_count']:,}")
    middle.metric("Columns", dataset["column_count"])
    right.metric(
        "Columns with gaps",
        sum(1 for column in dataset["columns"] if column["missing_count"]),
    )

    st.dataframe(
        pd.DataFrame(dataset["columns"]).rename(
            columns={
                "name": "Column",
                "dtype": "Type",
                "kind": "Kind",
                "missing_count": "Missing",
                "unique_count": "Unique",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if dataset["warnings"]:
        with st.expander(f"Data quality notes ({len(dataset['warnings'])})"):
            for warning in dataset["warnings"]:
                st.write(f"- {warning}")


def _conversation_panel(api: AnalystApi) -> None:
    st.subheader("Ask a question")

    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            _render_answer(entry["answer"])

    question = st.chat_input("e.g. Which category earns the most revenue?")
    if not question:
        return

    if not st.session_state.session_id:
        st.error("No active session. Start one from the sidebar.")
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            try:
                answer = api.ask(st.session_state.session_id, question)
            except ApiError as exc:
                st.error(str(exc))
                return
        _render_answer(answer)

    st.session_state.history.append({"question": question, "answer": answer})


def _render_answer(answer: dict[str, Any]) -> None:
    icon, label = KIND_LABELS.get(answer.get("kind", ""), ("💬", "Answer"))
    if answer.get("kind") != "answer":
        st.warning(f"{icon} {label}")
    st.write(answer.get("answer", ""))

    rows = (answer.get("result") or {}).get("rows") or []
    if rows:
        frame = pd.DataFrame(rows)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        _maybe_chart(frame)

    _render_details(answer)


def _maybe_chart(frame: pd.DataFrame) -> None:
    """Chart a result only when its shape is unambiguous.

    One label column and one numeric column is the common grouped result;
    anything else is left as a table rather than guessed at.
    """
    numeric = frame.select_dtypes("number").columns.tolist()
    labels = [name for name in frame.columns if name not in numeric]
    if len(numeric) == 1 and len(labels) == 1 and 1 < len(frame) <= 50:
        st.bar_chart(frame, x=labels[0], y=numeric[0])


def _render_details(answer: dict[str, Any]) -> None:
    steps = answer.get("steps") or []
    if not steps:
        return

    with st.expander("How this was worked out"):
        for step in steps:
            if step.get("error"):
                st.write(f"{step['number']}. `{step['tool']}` — {step['error']}")
            else:
                rows = step.get("row_count")
                suffix = f" → {rows} row(s)" if rows is not None else ""
                st.write(f"{step['number']}. `{step['tool']}`{suffix}")
                st.json(step.get("arguments") or {}, expanded=False)

        grounding = answer.get("grounding")
        if grounding and not grounding.get("is_grounded"):
            st.warning(
                "These figures were not found in the computed results: "
                + ", ".join(grounding.get("unsupported", []))
            )


if __name__ == "__main__":
    main()
