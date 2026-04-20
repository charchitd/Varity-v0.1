"""Google Gemini provider for Varity."""

from typing import Any

import httpx

from varity.exceptions import ProviderError, QuotaExceededError
from varity.providers.base import BaseLLMProvider

_DEFAULT_MODEL = "gemini-2.0-flash"
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(BaseLLMProvider):
    """LLM provider that calls the Google Gemini generateContent API.

    Uses raw :mod:`httpx` — no Google Gen AI SDK dependency.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        base_url: str = _BASE_URL,
        **kwargs: Any,
    ) -> None:
        """Initialise the Gemini provider.

        Args:
            api_key: Google AI Studio API key (BYOK — never logged).
            model: Model identifier. Defaults to ``gemini-2.0-flash``.
            base_url: Base URL for the generateContent endpoint.
            **kwargs: Forwarded to :class:`BaseLLMProvider`.
        """
        super().__init__(api_key=api_key, model=model, base_url=base_url, **kwargs)
        self._headers = {
            "content-type": "application/json",
        }

    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the Gemini generateContent API and return the reply.

        Args:
            prompt: User message content.
            system: Optional system instruction.

        Returns:
            The model's text response.

        Raises:
            ProviderError: On 401 (bad key), or any other non-retryable HTTP error.
        """
        url = f"{self.base_url}/{self.model}:generateContent"
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        async def _post() -> httpx.Response:
            response = await self._client.post(
                url,
                headers=self._headers,
                params={"key": self._api_key},
                json=payload,
            )
            if response.status_code == 401:
                raise ProviderError("Gemini: Invalid API key")

            if response.status_code == 429:
                try:
                    data = response.json()
                    err = data.get("error", {})
                    msg = err.get("message", "")
                    details = err.get("details", [])

                    # 1. Detect Hard Quota Exhaustion (RPD)
                    # Often contains "PerDay" or "limit: 0"
                    is_exhausted = "PerDay" in msg or "limit: 0" in msg
                    for d in details:
                        if d.get("@type") == "type.googleapis.com/google.rpc.QuotaFailure":
                            for v in d.get("violations", []):
                                if "PerDay" in v.get("quotaId", ""):
                                    is_exhausted = True
                                    break
                    
                    if is_exhausted:
                        raise QuotaExceededError(
                            "Gemini: Daily request quota exhausted. "
                            "Wait until tomorrow or switch to a paid plan / different model."
                        )

                    # 2. Extract retryDelay for temporary rate limits (RPM)
                    for d in details:
                        if "retryDelay" in d:
                            delay_str = d["retryDelay"].rstrip("s")
                            response.headers["Retry-After"] = str(int(float(delay_str)))
                            break
                except (ValueError, KeyError, QuotaExceededError):
                    # If it's a QuotaExceededError, re-raise it
                    raise
                except Exception:
                    pass

            response.raise_for_status()
            return response

        try:
            response = await self._with_retry(_post)
        except QuotaExceededError:
            # Fatal quota issue: do not wrap, propagate as-is
            raise
        except httpx.HTTPStatusError as exc:
            # If we still fail after retries, it's a persistent rate limit
            if exc.response.status_code == 429:
                raise ProviderError(
                    "Gemini: Quota exhausted or rate limit hit. "
                    "Try again in a minute or switch to a paid plan."
                ) from exc
            raise ProviderError(
                f"Gemini: HTTP {exc.response.status_code} — {exc.response.text}"
            ) from exc



        data = response.json()
        return str(data["candidates"][0]["content"]["parts"][0]["text"])
