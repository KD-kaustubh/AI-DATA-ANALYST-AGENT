"""The Gemini implementation of LLMClient.

This is the only module that imports a provider SDK. The import happens when
a client is built, not when the package loads, so the rest of the project
works without the SDK installed.
"""

from __future__ import annotations

from typing import Any

from analyst.errors import LLMConfigurationError, LLMProviderError, LLMResponseError
from analyst.llm import LLMConfig, load_config

# Analysis should be reproducible, so ask for the least creative output.
TEMPERATURE = 0.0


class GeminiClient:
    """Talks to Google's Gemini models."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or load_config()
        self._genai, self._types = _import_sdk()
        try:
            self._client = self._genai.Client(api_key=self.config.api_key)
        except Exception as exc:
            raise LLMConfigurationError(
                f"Could not start the Gemini client for model "
                f"'{self.config.model}'."
            ) from exc

    def generate(
        self, prompt: str, *, system: str | None = None, json_output: bool = False
    ) -> str:
        """Send one prompt and return the reply text."""
        settings = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=TEMPERATURE,
            response_mime_type="application/json" if json_output else None,
        )
        try:
            response = self._client.models.generate_content(
                model=self.config.model, contents=prompt, config=settings
            )
        except Exception as exc:
            # Provider errors can carry request details, so report the type
            # only. The original stays on the traceback for debugging.
            raise LLMProviderError(
                f"The request to model '{self.config.model}' failed "
                f"({type(exc).__name__})."
            ) from exc

        text = getattr(response, "text", None)
        if not text or not text.strip():
            raise LLMResponseError(
                f"Model '{self.config.model}' returned an empty response."
            )
        return text


def _import_sdk() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise LLMConfigurationError(
            "The google-genai package is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc
    return genai, types
