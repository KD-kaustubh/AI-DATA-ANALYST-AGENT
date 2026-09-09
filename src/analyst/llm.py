"""The language-model boundary.

Everything above this module talks to `LLMClient`, never to a provider SDK,
so swapping providers means adding one class. Provider-specific code lives in
`gemini.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

from analyst.errors import LLMConfigurationError

DEFAULT_MODEL = "gemini-2.5-flash"


@runtime_checkable
class LLMClient(Protocol):
    """What the rest of the project needs from a language model."""

    def generate(
        self, prompt: str, *, system: str | None = None, json_output: bool = False
    ) -> str:
        """Return the model's reply as text.

        `json_output` asks the provider for JSON when it supports that; a
        provider that does not may ignore it, so callers still parse
        defensively.
        """
        ...


@dataclass(frozen=True)
class LLMConfig:
    """Credentials and model choice, read from the environment."""

    # Kept out of repr so the key cannot reach a log line or a traceback.
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL


def load_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    """Read GOOGLE_API_KEY and MODEL_NAME.

    With no `env` given, a local .env file is loaded first and the process
    environment is used. Tests pass an explicit mapping instead.
    """
    if env is None:
        _load_dotenv()
        env = os.environ

    api_key = (env.get("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise LLMConfigurationError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    model = (env.get("MODEL_NAME") or "").strip() or DEFAULT_MODEL
    return LLMConfig(api_key=api_key, model=model)


def _load_dotenv() -> None:
    """Load a .env file if python-dotenv is installed; ignore it otherwise."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
