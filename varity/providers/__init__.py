"""Provider factory and registry for Varity."""

from __future__ import annotations

from typing import Any

from varity.exceptions import ConfigError
from varity.providers.anthropic import AnthropicProvider
from varity.providers.base import BaseLLMProvider
from varity.providers.gemini import GeminiProvider
from varity.providers.openai import OpenAIProvider

PROVIDER_MAP: dict[str, type[BaseLLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}


def get_provider(name: str, api_key: str, **kwargs: Any) -> BaseLLMProvider:
    """Instantiate a provider by name.

    Args:
        name: One of ``"anthropic"``, ``"openai"``, or ``"gemini"``.
        api_key: User-supplied API key (BYOK).
        **kwargs: Forwarded to the provider constructor (e.g. ``model``).

    Returns:
        A concrete :class:`~varity.providers.base.BaseLLMProvider` instance.

    Raises:
        ConfigError: If *name* is not a known provider.
    """
    cls = PROVIDER_MAP.get(name.lower())
    if cls is None:
        known = ", ".join(sorted(PROVIDER_MAP))
        raise ConfigError(
            f"Unknown provider '{name}'. Known providers: {known}"
        )
    return cls(api_key=api_key, **kwargs)


__all__ = [
    "PROVIDER_MAP",
    "get_provider",
    "BaseLLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
]
