"""Turn pandas and numpy values into plain Python.

Results travel to JSON, so numpy scalars and NaT/NaN must not leak out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def to_python(value: Any) -> Any:
    """Convert a numpy or pandas value into a plain Python equivalent.

    Missing values become None and timestamps become ISO strings. Lists and
    dicts are converted element by element.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [to_python(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_python(item) for key, item in value.items()}
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def to_optional_float(value: Any) -> float | None:
    """Convert a number to a float, mapping missing values to None."""
    if value is None or pd.isna(value):
        return None
    return float(value)
