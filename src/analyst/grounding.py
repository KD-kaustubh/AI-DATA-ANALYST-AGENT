"""A check that the numbers in an answer came from the evidence.

This is a safety net, not a proof. It flags figures that appear in an answer
but nowhere in the verified results, which catches the common case of a model
inventing or miscalculating a value. It cannot judge whether the wording of
an answer is a fair reading of the data.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

# Numbers as they appear in prose: 1234, 1,234.5, -12.75, 33.33%.
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Figures too common to be worth checking; they are usually ordinals or
# counts of things said in the sentence rather than claims about the data.
_IGNORED = frozenset({0.0, 1.0, 2.0})


@dataclass(frozen=True)
class GroundingReport:
    """Which numbers in an answer could be traced back to the evidence."""

    is_grounded: bool
    checked: list[str]
    unsupported: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_grounding(text: str, results: Iterable[dict[str, Any]]) -> GroundingReport:
    """Compare the numbers in `text` against those in the verified results."""
    known = _collect_numbers(list(results))

    checked: list[str] = []
    unsupported: list[str] = []
    for written in _NUMBER.findall(text or ""):
        claim = _as_number(written)
        if claim is None or claim in _IGNORED:
            continue
        checked.append(written)
        if not _supported(claim, written, known):
            unsupported.append(written)

    return GroundingReport(
        is_grounded=not unsupported, checked=checked, unsupported=unsupported
    )


def _supported(claim: float, written: str, known: set[float]) -> bool:
    """True when a claimed figure matches an evidence figure.

    Rounding is allowed in the direction a writer would round: 209.15 may be
    reported as 209.2 or 209, but 210 is not a match.
    """
    decimals = len(written.split(".")[1]) if "." in written else 0
    return any(
        abs(value - claim) < 1e-9 or round(value, decimals) == claim for value in known
    )


def _collect_numbers(payload: Any, found: set[float] | None = None) -> set[float]:
    """Every number anywhere in the evidence, including inside strings."""
    if found is None:
        found = set()

    if isinstance(payload, dict):
        for value in payload.values():
            _collect_numbers(value, found)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _collect_numbers(value, found)
    elif isinstance(payload, bool):
        pass
    elif isinstance(payload, (int, float)):
        found.add(float(payload))
    elif isinstance(payload, str):
        for written in _NUMBER.findall(payload):
            number = _as_number(written)
            if number is not None:
                found.add(number)
    return found


def _as_number(written: str) -> float | None:
    try:
        return float(written.replace(",", ""))
    except ValueError:
        return None
