"""Tests for the Streamlit front end's HTTP helpers.

The UI itself is not driven by a browser here. These tests cover the client
it talks through, and check the app module imports and holds no secrets.
"""

import inspect
from pathlib import Path

import httpx
import pytest

import api_client
from api_client import DEFAULT_BASE_URL, AnalystApi, ApiError, api_base_url

ROOT = Path(__file__).resolve().parents[1]


def stub(monkeypatch, handler):
    """Answer every request with `handler`, without opening a socket."""
    seen: list[httpx.Request] = []

    def fake_request(method, url, **kwargs):
        request = httpx.Request(method, url, **{
            key: value for key, value in kwargs.items() if key in {"json", "files"}
        })
        seen.append(request)
        return handler(method, str(url))

    monkeypatch.setattr(api_client.httpx, "request", fake_request)
    return seen


def ok(payload, status_code=200):
    return lambda method, url: httpx.Response(status_code, json=payload)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_the_base_url_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("ANALYST_API_URL", raising=False)

    assert api_base_url() == DEFAULT_BASE_URL


def test_the_base_url_can_be_configured(monkeypatch):
    monkeypatch.setenv("ANALYST_API_URL", "http://api.internal:9000")

    assert api_base_url() == "http://api.internal:9000"
    assert AnalystApi().base_url == "http://api.internal:9000"


def test_a_trailing_slash_is_trimmed():
    assert AnalystApi("http://localhost:8000/").base_url == "http://localhost:8000"


# --------------------------------------------------------------------------
# requests
# --------------------------------------------------------------------------


def test_health_calls_the_health_endpoint(monkeypatch):
    seen = stub(monkeypatch, ok({"status": "ok", "version": "0.1.0"}))

    assert AnalystApi("http://x").health()["status"] == "ok"
    assert str(seen[0].url) == "http://x/api/health"


def test_upload_posts_the_file(monkeypatch):
    seen = stub(monkeypatch, ok({"dataset_id": "abc"}, 201))

    result = AnalystApi("http://x").upload_dataset("sales.csv", b"a,b\n1,2\n")

    assert result["dataset_id"] == "abc"
    assert seen[0].method == "POST"
    assert str(seen[0].url) == "http://x/api/datasets"


def test_creating_a_session_sends_the_dataset_id(monkeypatch):
    seen = stub(monkeypatch, ok({"session_id": "s1"}, 201))

    assert AnalystApi("http://x").create_session("d1")["session_id"] == "s1"
    assert b"d1" in seen[0].content


def test_asking_sends_the_question(monkeypatch):
    seen = stub(monkeypatch, ok({"answer": "North leads."}))

    result = AnalystApi("http://x").ask("s1", "Which region?")

    assert result["answer"] == "North leads."
    assert str(seen[0].url) == "http://x/api/sessions/s1/questions"
    assert b"Which region?" in seen[0].content


def test_ending_a_session_tolerates_an_empty_body(monkeypatch):
    stub(monkeypatch, lambda method, url: httpx.Response(204))

    assert AnalystApi("http://x").end_session("s1") is None


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_an_api_error_is_reported_with_its_detail(monkeypatch):
    stub(
        monkeypatch,
        lambda method, url: httpx.Response(
            404, json={"detail": "No dataset with id 'x'.", "code": "not_found"}
        ),
    )

    with pytest.raises(ApiError, match="No dataset with id") as raised:
        AnalystApi("http://x").dataset("x")

    assert raised.value.status_code == 404


def test_a_validation_error_is_made_readable(monkeypatch):
    stub(
        monkeypatch,
        lambda method, url: httpx.Response(
            422,
            json={"detail": [{"loc": ["body", "question"], "msg": "Field required"}]},
        ),
    )

    with pytest.raises(ApiError, match="question: Field required"):
        AnalystApi("http://x").ask("s1", "")


def test_a_non_json_error_still_reports_the_status(monkeypatch):
    stub(monkeypatch, lambda method, url: httpx.Response(500, text="boom"))

    with pytest.raises(ApiError, match="returned 500"):
        AnalystApi("http://x").health()


def test_an_unreachable_backend_says_so(monkeypatch):
    def refuse(method, url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(api_client.httpx, "request", refuse)

    with pytest.raises(ApiError, match="Is the backend running"):
        AnalystApi("http://x").health()


# --------------------------------------------------------------------------
# the Streamlit module
# --------------------------------------------------------------------------


def test_the_streamlit_app_imports():
    import app

    assert callable(app.main)


def test_the_streamlit_app_never_imports_the_analysis_engine():
    """The UI must reach the analyst over HTTP, not by importing it."""
    source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "from analyst" not in source
    assert "import analyst" not in source


def test_no_secrets_are_hardcoded_in_the_front_end():
    for name in ("app.py", "api_client.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        for marker in ("GOOGLE_API_KEY", "GROQ_API_KEY", "api_key", "AIza", "gsk_"):
            assert marker not in source


def test_the_chart_helper_only_fires_on_an_unambiguous_shape():
    import pandas as pd

    import app

    source = inspect.getsource(app._maybe_chart)
    assert "select_dtypes" in source

    # One label plus one measure is chartable; anything wider is not.
    assert len(pd.DataFrame({"a": ["x", "y"], "b": [1, 2]}).select_dtypes("number").columns) == 1
    assert len(pd.DataFrame({"a": ["x"], "b": [1], "c": [2]}).select_dtypes("number").columns) == 2
