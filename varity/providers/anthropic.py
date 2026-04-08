"""Anthropic Claude provider for Varity."""

from typing import Any

import httpx

from varity.exceptions import ProviderError
from varity.providers.base import BaseLLMProvider

_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_BASE_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_MAX_TOKENS = 4096


class AnthropicProvider(BaseLLMProvider):
    """LLM provider that calls the Anthropic Messages API.

    Uses raw :mod:`httpx` — no Anthropic SDK dependency.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        base_url: str = _BASE_URL,
        **kwargs: Any,
    ) -> None:
        """Initialise the Anthropic provider.

        Args:
            api_key: Anthropic API key (BYOK — never logged).
            model: Model identifier. Defaults to ``claude-sonnet-4-20250514``.
            base_url: Messages API URL. Defaults to the official endpoint.
            **kwargs: Forwarded to :class:`BaseLLMProvider`.
        """
        super().__init__(api_key=api_key, model=model, base_url=base_url, **kwargs)
        self._headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the Anthropic Messages API and return the reply.

        Args:
            prompt: User message content.
            system: Optional system prompt.

        Returns:
            The assistant's text response.

        Raises:
            ProviderError: On 401 (bad key), or any other non-retryable HTTP error.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        async def _post() -> httpx.Response:
            response = await self._client.post(
                self.base_url,
                headers=self._headers,
                json=payload,
            )
            if response.status_code == 401:
                raise ProviderError("Anthropic: Invalid API key")
            response.raise_for_status()
            return response

        try:
            response = await self._with_retry(_post)
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Anthropic: HTTP {exc.response.status_code} — {exc.response.text}"
            ) from exc

        data = response.json()
        return str(data["content"][0]["text"])
