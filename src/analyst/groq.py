"""The Groq implementation of LLMClient.

Like `gemini.py`, this is the only module that imports its provider SDK, and
it does so when a client is built rather than when the package loads.
"""

from __future__ import annotations

from typing import Any

from analyst.errors import LLMConfigurationError, LLMProviderError, LLMResponseError
from analyst.llm import GROQ, LLMConfig, load_config

# Analysis should be reproducible, so ask for the least creative output.
TEMPERATURE = 0.0


class GroqClient:
    """Talks to models hosted on Groq."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or load_config(provider=GROQ)
        sdk = _import_sdk()
        try:
            self._client = sdk.Groq(api_key=self.config.api_key)
        except Exception as exc:
            raise LLMConfigurationError(
                f"Could not start the Groq client for model "
                f"'{self.config.model}'."
            ) from exc

    def generate(
        self, prompt: str, *, system: str | None = None, json_output: bool = False
    ) -> str:
        """Send one prompt and return the reply text."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        settings: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": TEMPERATURE,
        }
        if json_output:
            settings["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**settings)
        except Exception as exc:
            # Provider errors can carry request details, so report the type
            # and status code only. The original stays on the traceback for
            # debugging. groq's APIStatusError exposes it as `.status_code`.
            status_code = _numeric_status(exc, "status_code")
            prefix = f"{status_code} " if status_code is not None else ""
            raise LLMProviderError(
                f"The request to model '{self.config.model}' failed "
                f"({prefix}{type(exc).__name__}).",
                status_code=status_code,
            ) from exc

        text = _first_message(response)
        if not text or not text.strip():
            raise LLMResponseError(
                f"Model '{self.config.model}' returned an empty response."
            )
        return text


def _first_message(response: Any) -> str | None:
    """Pull the reply text out of a chat completion."""
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    return getattr(getattr(choices[0], "message", None), "content", None)


def _numeric_status(exc: Exception, attribute: str) -> int | None:
    """Read an integer status code off an SDK exception, if it has one."""
    value = getattr(exc, attribute, None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _import_sdk() -> Any:
    try:
        import groq
    except ImportError as exc:
        raise LLMConfigurationError(
            "The groq package is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return groq
