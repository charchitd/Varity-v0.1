"""Utility helpers for Varity: token counting, cost estimation, logging."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Coroutine, TypeVar

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a named logger with a sensible default format.

    Args:
        name: Logger name, typically ``__name__`` of the caller.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

# Rough per-model token costs in USD per 1K tokens (input, output)
_COST_TABLE: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.005, 0.015),
    "gemini-2.0-flash": (0.0, 0.0),  # free tier
}

_CHARS_PER_TOKEN = 4  # fallback when tiktoken unavailable


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Estimate token count for *text* using tiktoken when available.

    Falls back to a character-based heuristic (4 chars ≈ 1 token) when
    tiktoken does not support the requested model.

    Args:
        text: The text to count tokens for.
        model: Model name used to select the correct tokeniser.

    Returns:
        Estimated token count (int).
    """
    try:
        import tiktoken  # optional dependency

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o-mini",
) -> float:
    """Estimate USD cost for a provider call.

    Args:
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        model: Model name used to look up per-token pricing.

    Returns:
        Estimated cost in USD as a float.
    """
    in_rate, out_rate = _COST_TABLE.get(model, (0.001, 0.001))
    return (input_tokens / 1000) * in_rate + (output_tokens / 1000) * out_rate


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def now_ms() -> int:
    """Return the current monotonic time in milliseconds.

    Returns:
        Current time as integer milliseconds.
    """
    return int(time.monotonic() * 1000)


# ---------------------------------------------------------------------------
# Type vars for the async retry decorator
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Coroutine[Any, Any, _T]]], Callable[..., Coroutine[Any, Any, _T]]]:
    """Decorator: retry an async function with exponential back-off.

    Args:
        max_attempts: Maximum number of attempts (including the first).
        base_delay: Seconds to wait before the first retry; doubles each attempt.
        exceptions: Exception types that trigger a retry.

    Returns:
        Decorator that wraps an async callable with retry logic.
    """
    import asyncio
    import functools

    def decorator(
        fn: Callable[..., Coroutine[Any, Any, _T]],
    ) -> Callable[..., Coroutine[Any, Any, _T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> _T:
            delay = base_delay
            last_exc: Exception = RuntimeError("No attempts made")
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        await asyncio.sleep(delay)
                        delay *= 2.0
            raise last_exc

        return wrapper

    return decorator
