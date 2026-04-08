"""Base LLM provider abstract class for Varity."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx

from varity.exceptions import DecompositionError

logger = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503}
_RETRY_DELAYS = [1.0, 2.0, 4.0]


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers.

    Subclasses must implement :meth:`complete`. All network I/O is async
    and performed through a shared :class:`httpx.AsyncClient`.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialise the provider.

        Args:
            api_key: The user-supplied API key. Never logged.
            model: Model identifier string passed to the provider API.
            base_url: Root URL for API requests.
            **kwargs: Additional provider-specific options (ignored by base).
        """
        self._api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=30.0)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the LLM and return the text completion.

        Args:
            prompt: The user prompt to send.
            system: Optional system instruction.

        Returns:
            The model's text response.

        Raises:
            ProviderError: On API or network failures.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        schema: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Call :meth:`complete` and parse the response as JSON.

        Args:
            prompt: The user prompt to send.
            system: Optional system instruction.
            schema: Unused JSON schema hint (reserved for future validation).

        Returns:
            Parsed JSON object as a Python dict.

        Raises:
            DecompositionError: If the response is not valid JSON.
            ProviderError: On API or network failures.
        """
        raw = await self.complete(prompt, system=system)
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove opening fence (e.g. ```json) and closing fence
            text = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
        try:
            result: dict[str, Any] = json.loads(text)
            return result
        except json.JSONDecodeError as exc:
            raise DecompositionError(
                f"Provider response was not valid JSON: {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the underlying HTTP client.

        Returns:
            None
        """
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BaseLLMProvider":
        """Enter the async context manager.

        Returns:
            Self.
        """
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit the async context manager and close the client.

        Args:
            *args: Exception info (unused).
        """
        await self.close()

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------

    @staticmethod
    async def _with_retry(coro_fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute an async callable with exponential-backoff retry.

        Retries on :class:`httpx.HTTPStatusError` with status codes in
        ``{429, 500, 502, 503}``. Raises the final error after 3 attempts.

        Args:
            coro_fn: Async callable to execute.
            *args: Positional arguments forwarded to *coro_fn*.
            **kwargs: Keyword arguments forwarded to *coro_fn*.

        Returns:
            The return value of *coro_fn*.

        Raises:
            httpx.HTTPStatusError: If all retries are exhausted.
        """
        last_exc: Optional[Exception] = None
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                return await coro_fn(*args, **kwargs)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in _RETRY_STATUSES:
                    last_exc = exc
                    logger.warning(
                        "Retryable HTTP %d on attempt %d/%d — waiting %.0fs",
                        exc.response.status_code,
                        attempt,
                        len(_RETRY_DELAYS),
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
        raise last_exc  # type: ignore[misc]
