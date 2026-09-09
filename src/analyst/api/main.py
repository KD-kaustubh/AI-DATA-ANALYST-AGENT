"""The HTTP layer.

Routes stay thin: they validate the request, call the existing analyst
services, and shape the reply. No analysis, prompting or provider handling
happens here.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Depends, FastAPI, File, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from analyst import __version__
from analyst.agent import answer_question
from analyst.api.schemas import (
    AnalyzeRequest,
    AnswerResponse,
    DatasetSummary,
    ErrorResponse,
    HealthResponse,
    QuestionRequest,
    SessionRequest,
    SessionResponse,
    answer_response,
    dataset_summary,
    session_response,
)
from analyst.api.store import Store, UnknownResource
from analyst.errors import (
    AnalysisError,
    DatasetError,
    LLMConfigurationError,
    LLMError,
    ToolError,
)
from analyst.llm import LLMClient, create_client
from analyst.loader import load_dataset
from analyst.validation import SUPPORTED_EXTENSIONS

# Uploads are held in memory while they are parsed, so keep them modest.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Where a local Streamlit app runs. Override with CORS_ORIGINS.
DEFAULT_ORIGINS = ("http://localhost:8501", "http://127.0.0.1:8501")

ClientFactory = Callable[[], LLMClient]


def create_app(
    *,
    client_factory: ClientFactory = create_client,
    store: Store | None = None,
    allowed_origins: list[str] | None = None,
) -> FastAPI:
    """Build the application.

    `client_factory` is called only when a question is asked, so the API
    imports and serves health checks without any provider key configured.
    Tests pass their own factory and store.
    """
    app = FastAPI(
        title="AI Data Analyst Agent",
        version=__version__,
        description=(
            "Ask questions about an uploaded dataset. Every number is "
            "computed by pandas, never by the language model."
        ),
    )
    app.state.store = store or Store()
    app.state.client_factory = client_factory

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or _configured_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    _register_error_handlers(app)
    app.include_router(_build_router(), prefix="/api")
    return app


def get_store(request: Request) -> Store:
    return request.app.state.store


def build_client(request: Request) -> LLMClient:
    """Build the model client for this request.

    Called from inside a handler rather than as a dependency, so an invalid
    request body is reported as 422 before any provider is touched. A missing
    or broken key surfaces as a configuration error, which the handler turns
    into 503 rather than a stack trace.
    """
    return request.app.state.client_factory()


def _build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        """Liveness check. Says nothing about configured credentials."""
        return HealthResponse(status="ok", version=__version__)

    @router.post(
        "/datasets",
        response_model=DatasetSummary,
        status_code=status.HTTP_201_CREATED,
        tags=["datasets"],
    )
    async def upload_dataset(
        file: UploadFile = File(...), store: Store = Depends(get_store)
    ) -> DatasetSummary:
        """Accept a CSV or XLSX file, validate and profile it."""
        payload = await file.read()
        suffix = _checked_suffix(file.filename)
        _check_size(payload)

        frame = _parse_upload(payload, suffix)
        return dataset_summary(store.add_dataset(_safe_name(file.filename), frame))

    @router.get(
        "/datasets/{dataset_id}", response_model=DatasetSummary, tags=["datasets"]
    )
    def get_dataset(
        dataset_id: str, store: Store = Depends(get_store)
    ) -> DatasetSummary:
        """The dataset's profile. Never its rows."""
        return dataset_summary(store.dataset(dataset_id))

    @router.post("/analyze", response_model=AnswerResponse, tags=["analysis"])
    def analyze(
        http_request: Request,
        body: AnalyzeRequest,
        store: Store = Depends(get_store),
    ) -> AnswerResponse:
        """Ask one question about a dataset, without keeping a conversation."""
        dataset = store.dataset(body.dataset_id)
        answer = answer_question(
            dataset.frame,
            body.question,
            build_client(http_request),
            profile=dataset.profile,
        )
        return answer_response(answer, dataset.dataset_id)

    @router.post(
        "/sessions",
        response_model=SessionResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["sessions"],
    )
    def create_session(
        http_request: Request,
        body: SessionRequest,
        store: Store = Depends(get_store),
    ) -> SessionResponse:
        """Open a conversation over an uploaded dataset."""
        # The dataset is checked first, so an unknown id is 404 rather than a
        # provider error from building a client we would not have used.
        store.dataset(body.dataset_id)
        return session_response(
            store.create_session(body.dataset_id, build_client(http_request))
        )

    @router.get(
        "/sessions/{session_id}", response_model=SessionResponse, tags=["sessions"]
    )
    def get_session(
        session_id: str, store: Store = Depends(get_store)
    ) -> SessionResponse:
        return session_response(store.session(session_id))

    @router.post(
        "/sessions/{session_id}/questions",
        response_model=AnswerResponse,
        tags=["sessions"],
    )
    def ask_session(
        session_id: str,
        body: QuestionRequest,
        store: Store = Depends(get_store),
    ) -> AnswerResponse:
        """Ask a question, using and extending the conversation."""
        session = store.session(session_id)
        answer = session.conversation.ask(body.question)
        return answer_response(answer, session.dataset_id, session.session_id)

    @router.delete(
        "/sessions/{session_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["sessions"],
    )
    def end_session(session_id: str, store: Store = Depends(get_store)) -> None:
        """End a session so the caller can start a fresh one."""
        store.drop_session(session_id)

    return router


def _register_error_handlers(app: FastAPI) -> None:
    """Map analyst errors onto HTTP responses, once, for every route."""

    def respond(code: str, detail: str, http_status: int) -> JSONResponse:
        return JSONResponse(
            status_code=http_status,
            content=ErrorResponse(detail=detail, code=code).model_dump(),
        )

    @app.exception_handler(UnknownResource)
    def _unknown(request: Request, exc: UnknownResource) -> JSONResponse:
        return respond("not_found", str(exc), status.HTTP_404_NOT_FOUND)

    @app.exception_handler(DatasetError)
    def _dataset(request: Request, exc: DatasetError) -> JSONResponse:
        return respond("invalid_dataset", str(exc), status.HTTP_400_BAD_REQUEST)

    @app.exception_handler(AnalysisError)
    def _analysis(request: Request, exc: AnalysisError) -> JSONResponse:
        return respond("invalid_request", str(exc), status.HTTP_400_BAD_REQUEST)

    @app.exception_handler(ToolError)
    def _tool(request: Request, exc: ToolError) -> JSONResponse:
        return respond("invalid_request", str(exc), status.HTTP_400_BAD_REQUEST)

    @app.exception_handler(LLMConfigurationError)
    def _configuration(request: Request, exc: LLMConfigurationError) -> JSONResponse:
        return respond(
            "provider_unconfigured",
            "The analyst is not configured with a model provider.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.exception_handler(LLMError)
    def _provider(request: Request, exc: LLMError) -> JSONResponse:
        # Phase 3 already strips provider detail from these messages.
        if getattr(exc, "status_code", None) == 429:
            return respond(
                "rate_limited",
                "The model provider is rate-limiting requests right now. "
                "Please try again shortly.",
                status.HTTP_429_TOO_MANY_REQUESTS,
            )
        return respond("provider_error", str(exc), status.HTTP_502_BAD_GATEWAY)

    @app.exception_handler(Exception)
    def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Never let an unplanned exception reach the client as a traceback.
        return respond(
            "internal_error",
            "The analyst failed to handle that request.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def _checked_suffix(filename: str | None) -> str:
    """Validate the extension without ever using the name as a path."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise DatasetError(
            f"Unsupported file type '{suffix or filename or 'unnamed'}'. "
            f"Supported types: {supported}"
        )
    return suffix


def _check_size(payload: bytes) -> None:
    if not payload:
        raise DatasetError("The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        limit = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise DatasetError(f"The uploaded file is larger than {limit} MB.")


def _parse_upload(payload: bytes, suffix: str):
    """Write the bytes to a temporary file and hand them to the loader.

    The name is generated here, never taken from the upload, and the file is
    removed as soon as it has been parsed.
    """
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        handle.write(payload)
        handle.close()
        return load_dataset(handle.name)
    finally:
        Path(handle.name).unlink(missing_ok=True)


def _safe_name(filename: str | None) -> str:
    """Keep the base name for display only, with any path stripped."""
    return Path(filename or "dataset").name[:120]


def _configured_origins() -> list[str]:
    configured = os.environ.get("CORS_ORIGINS", "").strip()
    if not configured:
        return list(DEFAULT_ORIGINS)
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app = create_app()
