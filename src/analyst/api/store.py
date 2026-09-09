"""In-memory storage for uploaded datasets and analysis sessions.

Everything lives in the server process, so restarting the API forgets every
dataset and session. That is deliberate for this phase. The whole store sits
behind one class, so a persistent implementation could replace it later
without touching the routes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from analyst.context import DatasetContext, build_context
from analyst.conversation import Conversation
from analyst.llm import LLMClient
from analyst.profiling import DatasetProfile, profile_dataset


class UnknownResource(LookupError):
    """A dataset or session id the store does not hold."""


@dataclass
class StoredDataset:
    """An uploaded dataset and the profile taken when it arrived."""

    dataset_id: str
    filename: str
    frame: pd.DataFrame
    profile: DatasetProfile
    created_at: datetime

    @property
    def context(self) -> DatasetContext:
        """The schema summary, built from the profile rather than the rows."""
        return build_context(self.profile)


@dataclass
class StoredSession:
    """A conversation bound to one dataset."""

    session_id: str
    dataset_id: str
    conversation: Conversation
    created_at: datetime


class Store:
    """Datasets and sessions held for the lifetime of the process."""

    def __init__(self) -> None:
        self._datasets: dict[str, StoredDataset] = {}
        self._sessions: dict[str, StoredSession] = {}
        # Endpoints are sync, so uvicorn runs them on a thread pool.
        self._lock = threading.Lock()

    def add_dataset(self, filename: str, frame: pd.DataFrame) -> StoredDataset:
        """Profile a frame once and keep it under a new id."""
        stored = StoredDataset(
            dataset_id=uuid4().hex,
            filename=filename,
            frame=frame,
            profile=profile_dataset(frame),
            created_at=_now(),
        )
        with self._lock:
            self._datasets[stored.dataset_id] = stored
        return stored

    def dataset(self, dataset_id: str) -> StoredDataset:
        with self._lock:
            stored = self._datasets.get(dataset_id)
        if stored is None:
            raise UnknownResource(f"No dataset with id '{dataset_id}'.")
        return stored

    def create_session(self, dataset_id: str, client: LLMClient) -> StoredSession:
        """Open a conversation over an already uploaded dataset."""
        dataset = self.dataset(dataset_id)
        session = StoredSession(
            session_id=uuid4().hex,
            dataset_id=dataset_id,
            conversation=Conversation(
                dataset.frame, client, profile=dataset.profile
            ),
            created_at=_now(),
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def session(self, session_id: str) -> StoredSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise UnknownResource(f"No session with id '{session_id}'.")
        return session

    def drop_session(self, session_id: str) -> None:
        """End a session, so a caller can start over."""
        with self._lock:
            if self._sessions.pop(session_id, None) is None:
                raise UnknownResource(f"No session with id '{session_id}'.")

    def clear(self) -> None:
        """Forget everything. Used between tests."""
        with self._lock:
            self._datasets.clear()
            self._sessions.clear()

    @property
    def dataset_count(self) -> int:
        return len(self._datasets)

    @property
    def session_count(self) -> int:
        return len(self._sessions)


def _now() -> datetime:
    return datetime.now(timezone.utc)
