"""Unit tests for RecursiveChecker and the public Varity API."""

from __future__ import annotations

import json

import pytest

from varity import Varity, VarityConfig
from varity.checker import RecursiveChecker
from varity.models import CheckResult
from varity.providers.base import BaseLLMProvider


# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------

class _MultiResponseProvider(BaseLLMProvider):
    """Returns different JSON payloads in sequence, then repeats the last."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(api_key="mock", model="mock")
        self._responses = responses
        self._index = 0

    async def complete(self, prompt: str, system: str = "") -> str:
        resp = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return resp


def _decompose_payload(n: int = 2) -> str:
    claims = [
        {"text": f"Claim {i}.", "claim_type": "factual", "source_span": [0, 8]}
        for i in range(n)
    ]
    return json.dumps({"claims": claims})


def _verify_payload(verdict: str = "supported") -> str:
    return json.dumps({"verdict": verdict, "reasoning": "OK.", "confidence_delta": 0.1})


def _make_provider(n_claims: int = 2, verdict: str = "supported") -> _MultiResponseProvider:
    """Build a provider that returns decompose JSON first, then verify/cross JSONs."""
    responses = [_decompose_payload(n_claims)]
    # n_claims × (depth+1 verify steps) + n_claims cross-check steps
    for _ in range(n_claims * 10):
        responses.append(_verify_payload(verdict))
    return _MultiResponseProvider(responses)


# ===========================================================================
# RecursiveChecker
# ===========================================================================

@pytest.mark.asyncio
async def test_checker_returns_check_result() -> None:
    provider = _make_provider(n_claims=2)
    checker = RecursiveChecker(provider, VarityConfig(depth=1, strategy="quick"))
    result = await checker.run("Some LLM response text.")
    assert isinstance(result, CheckResult)
    assert result.original_response == "Some LLM response text."


@pytest.mark.asyncio
async def test_checker_claims_populated() -> None:
    provider = _make_provider(n_claims=3)
    checker = RecursiveChecker(provider, VarityConfig(depth=1, strategy="quick"))
    result = await checker.run("text")
    assert len(result.claims) == 3


@pytest.mark.asyncio
async def test_checker_empty_decompose_returns_empty_result() -> None:
    provider = _MultiResponseProvider([json.dumps({"claims": []})])
    checker = RecursiveChecker(provider)
    result = await checker.run("text")
    assert result.claims == []
    assert result.overall_confidence == 1.0
    assert result.vss_score == 1.0


@pytest.mark.asyncio
async def test_checker_supported_claims_not_flagged() -> None:
    provider = _make_provider(n_claims=2, verdict="supported")
    checker = RecursiveChecker(provider, VarityConfig(depth=1, confidence_threshold=0.3))
    result = await checker.run("text")
    assert all(not c.flagged for c in result.claims)


@pytest.mark.asyncio
async def test_checker_contradicted_claims_flagged() -> None:
    provider = _make_provider(n_claims=2, verdict="contradicted")
    checker = RecursiveChecker(provider, VarityConfig(depth=1, confidence_threshold=0.5))
    result = await checker.run("text")
    assert all(c.flagged for c in result.claims)


@pytest.mark.asyncio
async def test_checker_duration_ms_positive() -> None:
    provider = _make_provider(n_claims=1)
    checker = RecursiveChecker(provider, VarityConfig(depth=0, strategy="quick"))
    result = await checker.run("text")
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_checker_corrected_response_generated_when_flagged() -> None:
    responses = [_decompose_payload(1)]
    for _ in range(20):
        responses.append(_verify_payload("contradicted"))
    responses.append("Corrected response text.")
    provider = _MultiResponseProvider(responses)

    checker = RecursiveChecker(
        provider,
        VarityConfig(depth=1, confidence_threshold=0.9, strategy="quick"),
    )
    result = await checker.run("text")
    if result.flagged_claims:
        assert result.corrected_response is not None


@pytest.mark.asyncio
async def test_checker_no_correction_when_nothing_flagged() -> None:
    provider = _make_provider(n_claims=2, verdict="supported")
    checker = RecursiveChecker(
        provider, VarityConfig(depth=1, confidence_threshold=0.0, strategy="quick")
    )
    result = await checker.run("text")
    assert result.corrected_response is None


@pytest.mark.asyncio
async def test_checker_verification_chain_not_empty() -> None:
    provider = _make_provider(n_claims=2)
    checker = RecursiveChecker(provider, VarityConfig(depth=1, strategy="quick"))
    result = await checker.run("text")
    assert len(result.verification_chain) > 0


# ===========================================================================
# Varity public API
# ===========================================================================

@pytest.mark.asyncio
async def test_varity_acheck_returns_check_result() -> None:
    provider = _make_provider(n_claims=1)
    varity = Varity(provider=provider, config=VarityConfig(depth=0, strategy="quick"))
    result = await varity.acheck("Hello world.")
    assert isinstance(result, CheckResult)


def test_varity_check_sync_runs() -> None:
    """check() must run without an existing event loop."""
    provider = _make_provider(n_claims=1)
    varity = Varity(provider=provider, config=VarityConfig(depth=0, strategy="quick"))
    result = varity.check("Hello world.")
    assert isinstance(result, CheckResult)


def test_varity_default_config() -> None:
    """Varity uses VarityConfig defaults when config is omitted."""
    from varity.providers.anthropic import AnthropicProvider
    provider = AnthropicProvider(api_key="test-key")
    varity = Varity(provider=provider)
    assert varity._checker._config.depth == 2


@pytest.mark.asyncio
async def test_varity_check_raises_in_running_loop() -> None:
    """check() raises RuntimeError when called inside a running event loop."""
    import pytest
    provider = _make_provider(n_claims=1)
    varity = Varity(provider=provider, config=VarityConfig(depth=0, strategy="quick"))
    with pytest.raises(RuntimeError, match="running event loop"):
        varity.check("Hello world.")


@pytest.mark.asyncio
async def test_token_usage_populated() -> None:
    """token_usage dict should have prompt/completion/total keys after a run."""
    provider = _make_provider(n_claims=2)
    checker = RecursiveChecker(provider, VarityConfig(depth=1, strategy="quick"))
    result = await checker.run("Some response text with claims.")
    assert "prompt_tokens" in result.token_usage
    assert "completion_tokens" in result.token_usage
    assert "total_tokens" in result.token_usage
    assert result.token_usage["total_tokens"] > 0
    assert (
        result.token_usage["total_tokens"]
        == result.token_usage["prompt_tokens"] + result.token_usage["completion_tokens"]
    )


@pytest.mark.asyncio
async def test_empty_response_token_usage_zero() -> None:
    """Empty response produces zero token_usage."""
    provider = _MultiResponseProvider([json.dumps({"claims": []})])
    checker = RecursiveChecker(provider)
    result = await checker.run("")
    assert result.token_usage.get("total_tokens") == 0
