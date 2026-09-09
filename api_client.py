"""HTTP client for the analyst API.

The Streamlit app talks to the backend only through this module, so the UI
never imports the analysis engine. Kept out of the `analyst` package because
it is a consumer of the API, not part of it.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 120.0


class ApiError(RuntimeError):
    """The API refused a request, or could not be reached."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AnalystApi:
    """A thin wrapper over the endpoints the UI needs."""

    def __init__(
        self, base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT
    ) -> None:
        self.base_url = (base_url or api_base_url()).rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/health")

    def upload_dataset(self, filename: str, payload: bytes) -> dict[str, Any]:
        return self._request(
            "POST", "/api/datasets", files={"file": (filename, payload)}
        )

    def dataset(self, dataset_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/datasets/{dataset_id}")

    def create_session(self, dataset_id: str) -> dict[str, Any]:
        return self._request("POST", "/api/sessions", json={"dataset_id": dataset_id})

    def ask(self, session_id: str, question: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/sessions/{session_id}/questions",
            json={"question": question},
        )

    def end_session(self, session_id: str) -> None:
        self._request("DELETE", f"/api/sessions/{session_id}")

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = httpx.request(
                method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
            )
        except httpx.HTTPError as exc:
            raise ApiError(
                f"Could not reach the analyst API at {self.base_url}. "
                "Is the backend running?"
            ) from exc

        if response.status_code >= 400:
            raise ApiError(_error_message(response), response.status_code)
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()


def api_base_url() -> str:
    """Where the backend lives. Override with ANALYST_API_URL."""
    return os.environ.get("ANALYST_API_URL", "").strip() or DEFAULT_BASE_URL


def _error_message(response: httpx.Response) -> str:
    """Read the API's error shape, falling back to the status line."""
    try:
        payload = response.json()
    except ValueError:
        return f"The API returned {response.status_code}."

    detail = payload.get("detail")
    if isinstance(detail, str) and detail:
        return detail
    if isinstance(detail, list) and detail:
        # A FastAPI validation error: report the first problem plainly.
        first = detail[0]
        field = ".".join(str(part) for part in first.get("loc", [])[1:])
        message = first.get("msg", "Invalid request")
        return f"{field}: {message}" if field else message
    return f"The API returned {response.status_code}."
