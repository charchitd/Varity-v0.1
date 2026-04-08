"""OpenAI provider for Varity."""

from typing import Any

import httpx

from varity.exceptions import ProviderError
from varity.providers.base import BaseLLMProvider

_DEFAULT_MODEL = "gpt-4o-mini"
_BASE_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIProvider(BaseLLMProvider):
    """LLM provider that calls the OpenAI Chat Completions API.

    Uses raw :mod:`httpx` — no OpenAI SDK dependency.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        base_url: str = _BASE_URL,
        **kwargs: Any,
    ) -> None:
        """Initialise the OpenAI provider.

        Args:
            api_key: OpenAI API key (BYOK — never logged).
            model: Model identifier. Defaults to ``gpt-4o-mini``.
            base_url: Chat completions URL. Defaults to the official endpoint.
            **kwargs: Forwarded to :class:`BaseLLMProvider`.
        """
        super().__init__(api_key=api_key, model=model, base_url=base_url, **kwargs)
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }

    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the OpenAI Chat Completions API and return the reply.

        Args:
            prompt: User message content.
            system: Optional system prompt prepended as a system message.

        Returns:
            The assistant's text response.

        Raises:
            ProviderError: On 401 (bad key), or any other non-retryable HTTP error.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        async def _post() -> httpx.Response:
            response = await self._client.post(
                self.base_url,
                headers=self._headers,
                json=payload,
            )
            if response.status_code == 401:
                raise ProviderError("OpenAI: Invalid API key")
            response.raise_for_status()
            return response

        try:
            response = await self._with_retry(_post)
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"OpenAI: HTTP {exc.response.status_code} — {exc.response.text}"
            ) from exc

        data = response.json()
        return str(data["choices"][0]["message"]["content"])
