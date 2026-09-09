"""Tests for the HTTP API.

The app is built with a stubbed client factory, so no provider is contacted
and no API key is needed.
"""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from analyst.api import Store, create_app
from analyst.errors import LLMConfigurationError, LLMProviderError
from conftest import FakeLLM
from test_agent import GROUP_CALL, clarify, done, tool_call

CSV = "region,revenue\nNorth,100.0\nSouth,50.0\nNorth,110.0\n"
ANSWER_SCRIPT = (GROUP_CALL, done(), "North leads with 210.")


def build(*replies, factory=None):
    """An app whose model is scripted, plus its client and store."""
    model = FakeLLM(*replies)
    app = create_app(
        client_factory=factory or (lambda: model),
        store=Store(),
        allowed_origins=["http://localhost:8501"],
    )
    return TestClient(app, raise_server_exceptions=False), model


@pytest.fixture
def client():
    api, _ = build(*ANSWER_SCRIPT)
    return api


def upload(api, name="sales.csv", payload=None):
    body = CSV.encode() if payload is None else payload
    return api.post("/api/datasets", files={"file": (name, body, "text/csv")})


def uploaded_id(api) -> str:
    return upload(api).json()["dataset_id"]


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------


def test_health_reports_ok(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


def test_health_needs_no_provider():
    def explode():
        raise AssertionError("no client should be built for a health check")

    api, _ = build(factory=explode)

    assert api.get("/api/health").status_code == 200


# --------------------------------------------------------------------------
# uploading a dataset
# --------------------------------------------------------------------------


def test_uploads_a_csv(client):
    response = upload(client)

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "sales.csv"
    assert body["row_count"] == 3
    assert body["column_count"] == 2
    assert body["numeric_columns"] == ["revenue"]
    assert [column["name"] for column in body["columns"]] == ["region", "revenue"]


def test_uploads_an_xlsx(client, tmp_path):
    path = tmp_path / "sales.xlsx"
    pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]}).to_excel(path, index=False)

    response = upload(client, "sales.xlsx", path.read_bytes())

    assert response.status_code == 201
    assert response.json()["row_count"] == 2


def test_the_upload_response_carries_no_rows(client):
    body = json.dumps(upload(client).json())

    assert "110.0" not in body
    assert "rows" not in body


def test_an_unsupported_extension_is_refused(client):
    response = upload(client, "notes.txt", b"a,b\n1,2\n")

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_dataset"
    assert "Unsupported file type" in response.json()["detail"]


def test_an_empty_upload_is_refused(client):
    response = upload(client, "empty.csv", b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_a_corrupt_spreadsheet_is_refused(client):
    response = upload(client, "broken.xlsx", b"this is not a spreadsheet")

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_dataset"


def test_a_header_only_csv_is_refused(client):
    response = upload(client, "headers.csv", b"region,revenue\n")

    assert response.status_code == 400


def test_an_oversized_upload_is_refused(client, monkeypatch):
    monkeypatch.setattr("analyst.api.main.MAX_UPLOAD_BYTES", 10)

    response = upload(client, "big.csv", CSV.encode())

    assert response.status_code == 400
    assert "larger than" in response.json()["detail"]


def test_a_dangerous_filename_is_not_used_as_a_path(client):
    response = upload(client, "../../evil.csv")

    assert response.status_code == 201
    assert response.json()["filename"] == "evil.csv"


def test_a_missing_file_field_is_a_validation_error(client):
    assert client.post("/api/datasets", data={}).status_code == 422


# --------------------------------------------------------------------------
# reading a dataset back
# --------------------------------------------------------------------------


def test_fetches_a_dataset_profile(client):
    dataset_id = uploaded_id(client)

    response = client.get(f"/api/datasets/{dataset_id}")

    assert response.status_code == 200
    assert response.json()["dataset_id"] == dataset_id


def test_an_unknown_dataset_is_404(client):
    response = client.get("/api/datasets/does-not-exist")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------


def test_creates_a_session(client):
    dataset_id = uploaded_id(client)

    response = client.post("/api/sessions", json={"dataset_id": dataset_id})

    assert response.status_code == 201
    assert response.json()["dataset_id"] == dataset_id
    assert response.json()["turn_count"] == 0


def test_a_session_needs_a_known_dataset(client):
    response = client.post("/api/sessions", json={"dataset_id": "nope"})

    assert response.status_code == 404


def test_a_session_request_needs_a_dataset_id(client):
    assert client.post("/api/sessions", json={}).status_code == 422


def test_asks_a_question_in_a_session(client):
    dataset_id = uploaded_id(client)
    session_id = client.post(
        "/api/sessions", json={"dataset_id": dataset_id}
    ).json()["session_id"]

    response = client.post(
        f"/api/sessions/{session_id}/questions",
        json={"question": "Which region earns most?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "North leads with 210."
    assert body["kind"] == "answer"
    assert body["tools_used"] == ["group_aggregate"]
    assert body["session_id"] == session_id
    assert body["result"]["rows"][0]["revenue_sum"] == 210.0
    assert body["grounding"]["is_grounded"] is True


def test_a_follow_up_uses_the_same_session():
    api, model = build(
        *ANSWER_SCRIPT,
        tool_call("sort_rows", by="revenue", ascending=False, limit=1),
        done(),
        "North has the single largest sale.",
    )
    dataset_id = uploaded_id(api)
    session_id = api.post("/api/sessions", json={"dataset_id": dataset_id}).json()[
        "session_id"
    ]
    api.post(f"/api/sessions/{session_id}/questions", json={"question": "Totals?"})

    second = api.post(
        f"/api/sessions/{session_id}/questions", json={"question": "Which is highest?"}
    )

    assert second.status_code == 200
    assert second.json()["tools_used"] == ["sort_rows"]
    assert api.get(f"/api/sessions/{session_id}").json()["turn_count"] == 2
    # The follow-up prompt carried the earlier turn.
    assert "Totals?" in model.calls[3]["prompt"]


def test_a_question_on_an_unknown_session_is_404(client):
    response = client.post(
        "/api/sessions/nope/questions", json={"question": "Anything?"}
    )

    assert response.status_code == 404


@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": "x" * 1001}])
def test_an_invalid_question_is_422(client, body):
    dataset_id = uploaded_id(client)
    session_id = client.post(
        "/api/sessions", json={"dataset_id": dataset_id}
    ).json()["session_id"]

    response = client.post(f"/api/sessions/{session_id}/questions", json=body)

    assert response.status_code == 422


def test_ends_a_session(client):
    dataset_id = uploaded_id(client)
    session_id = client.post(
        "/api/sessions", json={"dataset_id": dataset_id}
    ).json()["session_id"]

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


def test_ending_an_unknown_session_is_404(client):
    assert client.delete("/api/sessions/nope").status_code == 404


# --------------------------------------------------------------------------
# one-shot analysis
# --------------------------------------------------------------------------


def test_analyzes_without_a_session(client):
    dataset_id = uploaded_id(client)

    response = client.post(
        "/api/analyze",
        json={"dataset_id": dataset_id, "question": "Which region earns most?"},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] is None
    assert response.json()["tools_used"] == ["group_aggregate"]


def test_analyze_on_an_unknown_dataset_is_404(client):
    response = client.post(
        "/api/analyze", json={"dataset_id": "nope", "question": "Anything?"}
    )

    assert response.status_code == 404


def test_analyze_validates_its_body(client):
    assert client.post("/api/analyze", json={"question": "Hi"}).status_code == 422


# --------------------------------------------------------------------------
# clarification and unfinished analyses
# --------------------------------------------------------------------------


def test_a_clarification_comes_back_as_such():
    api, _ = build(clarify("Do you mean by revenue or by count?"))
    dataset_id = uploaded_id(api)

    response = api.post(
        "/api/analyze",
        json={"dataset_id": dataset_id, "question": "Show me the best one."},
    )

    body = response.json()
    assert body["kind"] == "clarification"
    assert body["answer"] == "Do you mean by revenue or by count?"
    assert body["tools_used"] == []


def test_an_analysis_error_is_reported_back_not_raised():
    api, _ = build(
        tool_call("value_counts", column="missing_column"),
        done("That column is not in the dataset."),
    )
    dataset_id = uploaded_id(api)

    response = api.post(
        "/api/analyze", json={"dataset_id": dataset_id, "question": "Counts?"}
    )

    assert response.status_code == 200
    assert response.json()["steps"][0]["error"]
    assert "missing_column" in response.json()["steps"][0]["error"]


# --------------------------------------------------------------------------
# provider problems
# --------------------------------------------------------------------------


def test_a_provider_failure_is_502_without_provider_detail():
    class Broken:
        def generate(self, prompt, *, system=None, json_output=False):
            raise LLMProviderError("The request to model 'x' failed (RuntimeError).")

    api, _ = build(factory=Broken)
    dataset_id = uploaded_id(api)

    response = api.post(
        "/api/analyze", json={"dataset_id": dataset_id, "question": "Totals?"}
    )

    assert response.status_code == 502
    assert response.json()["code"] == "provider_error"


def test_a_missing_key_is_503_and_says_nothing_about_keys():
    def unconfigured():
        raise LLMConfigurationError("GOOGLE_API_KEY is not set. Add your key.")

    api, _ = build(factory=unconfigured)
    dataset_id = uploaded_id(api)

    response = api.post(
        "/api/analyze", json={"dataset_id": dataset_id, "question": "Totals?"}
    )

    assert response.status_code == 503
    assert response.json()["code"] == "provider_unconfigured"
    assert "GOOGLE_API_KEY" not in response.text


def test_an_unexpected_failure_is_a_plain_500():
    class Exploding:
        def generate(self, prompt, *, system=None, json_output=False):
            raise ZeroDivisionError("secret-token-abc in the message")

    api, _ = build(factory=Exploding)
    dataset_id = uploaded_id(api)

    response = api.post(
        "/api/analyze", json={"dataset_id": dataset_id, "question": "Totals?"}
    )

    # The agent wraps unknown client failures as a provider error.
    assert response.status_code in (500, 502)
    assert "secret-token-abc" not in response.text
    assert "Traceback" not in response.text


# --------------------------------------------------------------------------
# responses stay clean
# --------------------------------------------------------------------------


def test_every_response_is_json_serialisable(client):
    dataset_id = uploaded_id(client)
    session_id = client.post(
        "/api/sessions", json={"dataset_id": dataset_id}
    ).json()["session_id"]

    for response in (
        client.get("/api/health"),
        client.get(f"/api/datasets/{dataset_id}"),
        client.get(f"/api/sessions/{session_id}"),
        client.post(
            f"/api/sessions/{session_id}/questions", json={"question": "Totals?"}
        ),
    ):
        assert json.loads(response.text) is not None


def test_no_prompt_or_key_material_reaches_the_client(client):
    dataset_id = uploaded_id(client)

    response = client.post(
        "/api/analyze", json={"dataset_id": dataset_id, "question": "Totals?"}
    )

    for leak in ("api_key", "GOOGLE_API_KEY", "GROQ_API_KEY", "system_instruction"):
        assert leak not in response.text
    # The agent's prompts must not be echoed back either.
    assert "Dataset schema:" not in response.text


def test_cors_allows_the_configured_origin(client):
    response = client.get(
        "/api/health", headers={"Origin": "http://localhost:8501"}
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"


def test_cors_does_not_allow_every_origin(client):
    response = client.get("/api/health", headers={"Origin": "http://evil.example"})

    assert response.headers.get("access-control-allow-origin") != "*"
