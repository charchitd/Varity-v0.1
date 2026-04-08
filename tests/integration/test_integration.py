"""Integration tests for Varity — require a real API key.

These tests make actual LLM API calls.  They are skipped automatically
unless the following environment variables are set:

    VARITY_API_KEY   — your API key
    VARITY_PROVIDER  — one of: anthropic | openai | gemini (default: anthropic)

Run with:
    pytest tests/integration/ -v

Or explicitly:
    VARITY_API_KEY=sk-ant-... pytest tests/integration/ -v
"""

from __future__ import annotations

import os

import pytest

from varity import Varity, VarityConfig
from varity.models import CheckResult
from varity.providers import get_provider

# ---------------------------------------------------------------------------
# Fixtures & skip guard
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("VARITY_API_KEY", "")
_PROVIDER = os.environ.get("VARITY_PROVIDER", "anthropic")

pytestmark = pytest.mark.skipif(
    not _API_KEY,
    reason="VARITY_API_KEY not set — skipping integration tests",
)

# A response with one clear factual error (visible from space myth) and
# one correct claim (Python 1991) — lets us assert directional results.
_TEST_RESPONSE = (
    "Python was created by Guido van Rossum and released in 1991. "
    "The Great Wall of China is clearly visible from space with the naked eye."
)

_CLEAN_RESPONSE = (
    "Python was created by Guido van Rossum and first released in 1991. "
    "Water boils at 100 degrees Celsius at standard atmospheric pressure."
)


@pytest.fixture(scope="module")
def provider() -> object:
    return get_provider(_PROVIDER, api_key=_API_KEY)


@pytest.fixture(scope="module")
def varity(provider: object) -> Varity:
    from varity.providers.base import BaseLLMProvider
    assert isinstance(provider, BaseLLMProvider)
    config = VarityConfig(depth=1, strategy="quick", confidence_threshold=0.5)
    return Varity(provider=provider, config=config)


# ===========================================================================
# Integration Test 1 — pipeline returns a CheckResult
# ===========================================================================

@pytest.mark.asyncio
async def test_pipeline_returns_check_result(varity: Varity) -> None:
    result = await varity.acheck(_TEST_RESPONSE)
    assert isinstance(result, CheckResult)


# ===========================================================================
# Integration Test 2 — claims are extracted
# ===========================================================================

@pytest.mark.asyncio
async def test_claims_are_extracted(varity: Varity) -> None:
    result = await varity.acheck(_TEST_RESPONSE)
    assert len(result.claims) >= 1, "Expected at least one claim to be extracted"


# ===========================================================================
# Integration Test 3 — known bad claim is flagged
# ===========================================================================

@pytest.mark.asyncio
async def test_known_bad_claim_flagged(varity: Varity) -> None:
    """The 'visible from space' myth should be flagged or have low confidence."""
    result = await varity.acheck(_TEST_RESPONSE)
    # Find the claim about the Great Wall being visible from space
    wall_claims = [
        c for c in result.claims
        if "wall" in c.text.lower() or "space" in c.text.lower() or "visible" in c.text.lower()
    ]
    assert wall_claims, "Expected a claim about the Great Wall / space visibility"
    wall_claim = wall_claims[0]
    # It should either be flagged or have low confidence (< 0.7)
    assert wall_claim.flagged or wall_claim.confidence < 0.7, (
        f"Expected flagged=True or confidence<0.7 for: '{wall_claim.text}' "
        f"(conf={wall_claim.confidence:.2f}, flagged={wall_claim.flagged})"
    )


# ===========================================================================
# Integration Test 4 — clean response has high overall confidence
# ===========================================================================

@pytest.mark.asyncio
async def test_clean_response_has_high_confidence(varity: Varity) -> None:
    result = await varity.acheck(_CLEAN_RESPONSE)
    assert result.overall_confidence > 0.4, (
        f"Expected overall_confidence > 0.4 for a clean response, "
        f"got {result.overall_confidence:.2f}"
    )


# ===========================================================================
# Integration Test 5 — VSS is in [0, 1]
# ===========================================================================

@pytest.mark.asyncio
async def test_vss_in_valid_range(varity: Varity) -> None:
    result = await varity.acheck(_TEST_RESPONSE)
    assert 0.0 <= result.vss_score <= 1.0


# ===========================================================================
# Integration Test 6 — verification chain is populated
# ===========================================================================

@pytest.mark.asyncio
async def test_verification_chain_populated(varity: Varity) -> None:
    result = await varity.acheck(_TEST_RESPONSE)
    assert len(result.verification_chain) > 0


# ===========================================================================
# Integration Test 7 — token usage is estimated
# ===========================================================================

@pytest.mark.asyncio
async def test_token_usage_estimated(varity: Varity) -> None:
    result = await varity.acheck(_TEST_RESPONSE)
    assert result.token_usage.get("total_tokens", 0) > 0


# ===========================================================================
# Integration Test 8 — duration is recorded
# ===========================================================================

@pytest.mark.asyncio
async def test_duration_ms_recorded(varity: Varity) -> None:
    result = await varity.acheck(_TEST_RESPONSE)
    assert result.duration_ms > 0


# ===========================================================================
# Integration Test 9 — corrected response generated when claims flagged
# ===========================================================================

@pytest.mark.asyncio
async def test_corrected_response_when_flagged(varity: Varity) -> None:
    result = await varity.acheck(_TEST_RESPONSE)
    if result.flagged_claims:
        assert result.corrected_response is not None
        assert len(result.corrected_response) > 0


# ===========================================================================
# Integration Test 10 — empty string returns empty result gracefully
# ===========================================================================

@pytest.mark.asyncio
async def test_empty_response_graceful(varity: Varity) -> None:
    result = await varity.acheck("")
    assert result.claims == []
    assert result.overall_confidence == 1.0
    assert result.vss_score == 1.0
