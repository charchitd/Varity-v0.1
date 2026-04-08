"""Unit tests for varity.utils."""

from __future__ import annotations

import pytest

from varity.utils import async_retry, count_tokens, estimate_cost, get_logger, now_ms


def test_get_logger_returns_logger() -> None:
    import logging
    logger = get_logger("varity.test")
    assert isinstance(logger, logging.Logger)


def test_get_logger_idempotent() -> None:
    l1 = get_logger("varity.test.x")
    l2 = get_logger("varity.test.x")
    assert l1 is l2


def test_count_tokens_positive() -> None:
    n = count_tokens("Hello, world!", model="gpt-4o-mini")
    assert n > 0


def test_count_tokens_empty_string() -> None:
    n = count_tokens("", model="gpt-4o-mini")
    assert n >= 0


def test_count_tokens_longer_text_more_tokens() -> None:
    short = count_tokens("Hi", model="gpt-4o-mini")
    long = count_tokens("Hi " * 100, model="gpt-4o-mini")
    assert long > short


def test_estimate_cost_zero_for_free_tier() -> None:
    cost = estimate_cost(1000, 1000, model="gemini-2.0-flash")
    assert cost == 0.0


def test_estimate_cost_positive_for_paid() -> None:
    cost = estimate_cost(1000, 1000, model="gpt-4o-mini")
    assert cost > 0.0


def test_estimate_cost_unknown_model_uses_fallback() -> None:
    cost = estimate_cost(1000, 1000, model="unknown-model-xyz")
    assert cost > 0.0


def test_now_ms_returns_int() -> None:
    t = now_ms()
    assert isinstance(t, int)
    assert t > 0


def test_now_ms_increases_over_time() -> None:
    import time
    t1 = now_ms()
    time.sleep(0.01)
    t2 = now_ms()
    assert t2 > t1


@pytest.mark.asyncio
async def test_async_retry_succeeds_first_attempt() -> None:
    call_count = 0

    @async_retry(max_attempts=3)
    async def fn() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await fn()
    assert result == "ok"
    assert call_count == 1


@pytest.mark.asyncio
async def test_async_retry_retries_on_exception() -> None:
    call_count = 0

    @async_retry(max_attempts=3, base_delay=0.0, exceptions=(ValueError,))
    async def fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("retry me")
        return "ok"

    result = await fn()
    assert result == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_async_retry_raises_after_exhaustion() -> None:
    @async_retry(max_attempts=2, base_delay=0.0, exceptions=(RuntimeError,))
    async def fn() -> None:
        raise RuntimeError("always fails")

    with pytest.raises(RuntimeError, match="always fails"):
        await fn()


@pytest.mark.asyncio
async def test_async_retry_does_not_catch_other_exceptions() -> None:
    @async_retry(max_attempts=3, base_delay=0.0, exceptions=(ValueError,))
    async def fn() -> None:
        raise TypeError("not retried")

    with pytest.raises(TypeError):
        await fn()
