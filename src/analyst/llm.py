"""The language-model boundary.

Everything above this module talks to `LLMClient`, never to a provider SDK,
so adding a provider means adding one class and one entry in
PROVIDER_SETTINGS. Provider-specific code lives in `gemini.py` and `groq.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

from analyst.errors import LLMConfigurationError

GEMINI = "gemini"
GROQ = "groq"
PROVIDERS = (GEMINI, GROQ)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# Optional: names the provider to use when both keys are present.
PROVIDER_VARIABLE = "LLM_PROVIDER"


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
class ProviderSettings:
    """Which environment variables configure one provider."""

    name: str
    key_variable: str
    model_variable: str
    default_model: str


PROVIDER_SETTINGS: dict[str, ProviderSettings] = {
    GEMINI: ProviderSettings(GEMINI, "GOOGLE_API_KEY", "MODEL_NAME", DEFAULT_MODEL),
    GROQ: ProviderSettings(
        GROQ, "GROQ_API_KEY", "GROQ_MODEL_NAME", DEFAULT_GROQ_MODEL
    ),
}


@dataclass(frozen=True)
class LLMConfig:
    """Credentials and model choice, read from the environment."""

    # Kept out of repr so the key cannot reach a log line or a traceback.
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    provider: str = GEMINI


def get_provider_settings(provider: str) -> ProviderSettings:
    """Look up a provider, refusing names we do not support."""
    settings = PROVIDER_SETTINGS.get(provider)
    if settings is None:
        raise LLMConfigurationError(
            f"Unknown provider '{provider}'. Supported: {', '.join(PROVIDERS)}"
        )
    return settings


def load_config(
    env: Mapping[str, str] | None = None, provider: str = GEMINI
) -> LLMConfig:
    """Read the API key and model name for one provider.

    With no `env` given, a local .env file is loaded first and the process
    environment is used. Tests pass an explicit mapping instead.
    """
    settings = get_provider_settings(provider)
    env = _environment(env)

    api_key = (env.get(settings.key_variable) or "").strip()
    if not api_key:
        raise LLMConfigurationError(
            f"{settings.key_variable} is not set. "
            f"Copy .env.example to .env and add your {settings.name} key."
        )

    model = (env.get(settings.model_variable) or "").strip() or settings.default_model
    return LLMConfig(api_key=api_key, model=model, provider=settings.name)


def resolve_provider(env: Mapping[str, str] | None = None) -> str:
    """Decide which provider to use.

    LLM_PROVIDER wins when it is set. Otherwise the provider is whichever one
    has a key, and Gemini wins when both do.
    """
    env = _environment(env)

    named = (env.get(PROVIDER_VARIABLE) or "").strip().lower()
    if named:
        get_provider_settings(named)
        return named

    available = [
        name
        for name, settings in PROVIDER_SETTINGS.items()
        if (env.get(settings.key_variable) or "").strip()
    ]
    if not available:
        variables = ", ".join(
            settings.key_variable for settings in PROVIDER_SETTINGS.values()
        )
        raise LLMConfigurationError(
            f"No API key found. Set one of: {variables}. "
            "Copy .env.example to .env and add a key."
        )
    return GEMINI if GEMINI in available else available[0]


def create_client(
    provider: str | None = None, env: Mapping[str, str] | None = None
) -> LLMClient:
    """Build the client for `provider`, or for whichever key is configured."""
    chosen = provider or resolve_provider(env)
    config = load_config(env, provider=chosen)

    # Imported here so neither provider SDK is needed to use the other.
    if chosen == GEMINI:
        from analyst.gemini import GeminiClient

        return GeminiClient(config)

    from analyst.groq import GroqClient

    return GroqClient(config)


def _environment(env: Mapping[str, str] | None) -> Mapping[str, str]:
    if env is not None:
        return env
    _load_dotenv()
    return os.environ


def _load_dotenv() -> None:
    """Load a .env file if python-dotenv is installed; ignore it otherwise."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
