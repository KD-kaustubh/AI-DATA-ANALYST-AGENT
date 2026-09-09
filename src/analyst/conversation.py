"""In-memory conversation state for one dataset.

A Conversation holds a frame, its profile and the turns asked so far, so a
follow-up like "which one is highest?" can be read against what came before.
Nothing is persisted: when the object goes, so does the state. The class is
the seam a web layer would sit on later, one Conversation per session.

History sent to the model is bounded twice over: only the last few turns are
included, and only the newest of those keeps its result rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from analyst.agent import DEFAULT_MAX_STEPS, Answer, answer_question
from analyst.errors import InvalidOperationError
from analyst.llm import LLMClient
from analyst.profiling import DatasetProfile, profile_dataset

# Turns of history offered to the model.
DEFAULT_HISTORY_TURNS = 3

# Fields kept when an older turn's results are summarised.
_SUMMARY_FIELDS = ("operation", "columns", "parameters", "row_count")


@dataclass(frozen=True)
class Turn:
    """One question and the answer it produced."""

    question: str
    answer: str
    kind: str
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Conversation:
    """A dataset, a model client, and the turns asked so far."""

    def __init__(
        self,
        frame: pd.DataFrame,
        client: LLMClient,
        *,
        profile: DatasetProfile | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        history_turns: int = DEFAULT_HISTORY_TURNS,
    ) -> None:
        if frame.empty or frame.columns.empty:
            raise InvalidOperationError("There is no data to analyse.")
        if history_turns < 0:
            raise InvalidOperationError(
                f"history_turns cannot be negative, got {history_turns}."
            )

        self.frame = frame
        self.client = client
        self.profile = profile or profile_dataset(frame)
        self.max_steps = max_steps
        self.history_turns = history_turns
        self.turns: list[Turn] = []

    def ask(self, question: str) -> Answer:
        """Answer a question, using and then extending the conversation."""
        answer = answer_question(
            self.frame,
            question,
            self.client,
            profile=self.profile,
            max_steps=self.max_steps,
            history=self.history(),
        )
        self.turns.append(
            Turn(
                question=answer.question,
                answer=answer.text,
                kind=answer.kind,
                results=answer.results,
            )
        )
        return answer

    def history(self) -> list[dict[str, Any]]:
        """The bounded context handed to the model.

        The newest turn keeps its result rows, because a follow-up usually
        refers to it. Older turns keep only what was computed, not the
        numbers, so the prompt cannot grow with the conversation.
        """
        if not self.history_turns:
            return []

        recent = self.turns[-self.history_turns :]
        newest = len(recent) - 1
        return [
            self._entry(turn, detailed=index == newest)
            for index, turn in enumerate(recent)
        ]

    def reset(self) -> None:
        """Forget every turn, keeping the dataset and its profile."""
        self.turns.clear()

    def _entry(self, turn: Turn, *, detailed: bool) -> dict[str, Any]:
        entry: dict[str, Any] = {"question": turn.question, "answer": turn.answer}
        if not turn.results:
            return entry
        entry["results" if detailed else "results_summary"] = (
            turn.results if detailed else [_summarise(r) for r in turn.results]
        )
        return entry


def _summarise(result: dict[str, Any]) -> dict[str, Any]:
    """Keep what an older result was, without its rows."""
    return {key: result[key] for key in _SUMMARY_FIELDS if key in result}
