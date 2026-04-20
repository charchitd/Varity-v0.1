"""Varity — Recursive Self-Checking for LLM Hallucination Reduction."""

from __future__ import annotations

import asyncio
from typing import Optional

from varity.checker import RecursiveChecker
from varity.models import CheckResult, Claim, VarityConfig
from varity.providers.base import BaseLLMProvider

__version__ = "0.1.10"
__all__ = ["Varity", "CheckResult", "Claim", "VarityConfig", "__version__"]


class Varity:
    """Public API entry point for Varity.

    Wraps :class:`~varity.checker.RecursiveChecker` with a synchronous
    :meth:`check` interface so callers do not need to manage an event loop.
    An async :meth:`acheck` is also provided for callers already inside an
    async context (e.g. FastAPI handlers, Jupyter notebooks).

    Example::

        from varity import Varity, VarityConfig
        from varity.providers import get_provider

        provider = get_provider("anthropic", api_key="sk-ant-...")
        varity = Varity(provider)
        result = varity.check("The Eiffel Tower was built in 1887.")
        print(result.overall_confidence, result.vss_score)
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        config: Optional[VarityConfig] = None,
    ) -> None:
        """Initialise Varity.

        Args:
            provider: A concrete LLM provider (e.g. AnthropicProvider).
                      Obtain one via :func:`varity.providers.get_provider`.
            config: Optional :class:`~varity.models.VarityConfig`.  Defaults
                    are used when omitted.
        """
        self._checker = RecursiveChecker(provider=provider, config=config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, response: str) -> CheckResult:
        """Run the full verification pipeline synchronously.

        Blocks the calling thread until the pipeline completes.  Use
        :meth:`acheck` inside an existing async context instead to avoid
        a ``RuntimeError``.

        Args:
            response: The LLM response text to verify.

        Returns:
            :class:`~varity.models.CheckResult` with confidence scores,
            flagged claims, VSS, and an optional corrected response.

        Raises:
            RuntimeError: If called from within a running event loop.
                          Use ``await varity.acheck(response)`` instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — safe to use asyncio.run()
            return asyncio.run(self.acheck(response))
        raise RuntimeError(
            "Varity.check() cannot be called from within a running event loop. "
            "Use 'await varity.acheck(response)' instead."
        )

    async def acheck(self, response: str) -> CheckResult:
        """Run the full verification pipeline asynchronously.

        Args:
            response: The LLM response text to verify.

        Returns:
            :class:`~varity.models.CheckResult`.
        """
        return await self._checker.run(response)
