"""Unit tests for all four Varity strategy classes."""

from __future__ import annotations

import json
from typing import Any

import pytest

from varity.models import Claim, VerificationStep
from varity.providers.base import BaseLLMProvider
from varity.strategies.claim_decompose import ClaimDecomposer
from varity.strategies.confidence import ConfidenceAggregator
from varity.strategies.cross_check import CrossChecker
from varity.strategies.self_verify import SelfVerifier


# ---------------------------------------------------------------------------
# Shared MockProvider
# ---------------------------------------------------------------------------

class MockProvider(BaseLLMProvider):
    def __init__(self, response: str = '{"ok": true}', **kwargs: Any) -> None:
        super().__init__(api_key="mock", model="mock", **kwargs)
        self._response = response

    async def complete(self, prompt: str, system: str = "") -> str:
        return self._response


def _claim(text: str = "Paris is in France.", ctype: str = "factual") -> Claim:
    return Claim(text=text, claim_type=ctype, source_span=(0, len(text)))  # type: ignore[arg-type]


# ===========================================================================
# ClaimDecomposer
# ===========================================================================

@pytest.mark.asyncio
async def test_decompose_returns_claims() -> None:
    payload = json.dumps({
        "claims": [
            {"text": "Paris is in France.", "claim_type": "factual", "source_span": [0, 19]},
            {"text": "Python was released in 1991.", "claim_type": "temporal", "source_span": [20, 46]},
        ]
    })
    decomposer = ClaimDecomposer(MockProvider(response=payload), max_claims=10)
    claims = await decomposer.decompose("Paris is in France. Python was released in 1991.")
    assert len(claims) == 2
    assert claims[0].claim_type == "factual"
    assert claims[1].claim_type == "temporal"


@pytest.mark.asyncio
async def test_decompose_empty_response_returns_empty() -> None:
    decomposer = ClaimDecomposer(MockProvider(), max_claims=5)
    claims = await decomposer.decompose("")
    assert claims == []


@pytest.mark.asyncio
async def test_decompose_bad_json_degrades_gracefully() -> None:
    decomposer = ClaimDecomposer(MockProvider(response="not json"), max_claims=5)
    claims = await decomposer.decompose("Some response text.")
    assert claims == []


@pytest.mark.asyncio
async def test_decompose_missing_claims_key_degrades() -> None:
    decomposer = ClaimDecomposer(MockProvider(response='{"result": []}'), max_claims=5)
    claims = await decomposer.decompose("text")
    assert claims == []


@pytest.mark.asyncio
async def test_decompose_respects_max_claims() -> None:
    items = [
        {"text": f"Claim {i}.", "claim_type": "factual", "source_span": [0, 8]}
        for i in range(20)
    ]
    payload = json.dumps({"claims": items})
    decomposer = ClaimDecomposer(MockProvider(response=payload), max_claims=5)
    claims = await decomposer.decompose("text")
    assert len(claims) == 5


@pytest.mark.asyncio
async def test_decompose_invalid_claim_type_defaults_to_factual() -> None:
    payload = json.dumps({
        "claims": [{"text": "Something.", "claim_type": "bogus", "source_span": [0, 10]}]
    })
    decomposer = ClaimDecomposer(MockProvider(response=payload), max_claims=5)
    claims = await decomposer.decompose("Something.")
    assert claims[0].claim_type == "factual"


@pytest.mark.asyncio
async def test_decompose_skips_malformed_items() -> None:
    payload = json.dumps({
        "claims": [
            None,
            {"text": "", "claim_type": "factual", "source_span": [0, 0]},
            {"text": "Valid claim.", "claim_type": "factual", "source_span": [0, 12]},
        ]
    })
    decomposer = ClaimDecomposer(MockProvider(response=payload), max_claims=5)
    claims = await decomposer.decompose("Valid claim.")
    assert len(claims) == 1
    assert claims[0].text == "Valid claim."


# ===========================================================================
# SelfVerifier
# ===========================================================================

def _verify_response(verdict: str = "supported") -> str:
    return json.dumps({"verdict": verdict, "reasoning": "OK.", "confidence_delta": 0.1})


@pytest.mark.asyncio
async def test_self_verifier_returns_depth_plus_one_steps() -> None:
    verifier = SelfVerifier(MockProvider(response=_verify_response()), depth=2)
    claim = _claim()
    _, steps = await verifier.verify_all([claim])
    assert len(steps) == 3  # depths 0, 1, 2


@pytest.mark.asyncio
async def test_self_verifier_empty_claims() -> None:
    verifier = SelfVerifier(MockProvider(), depth=1)
    claims, steps = await verifier.verify_all([])
    assert claims == []
    assert steps == []


@pytest.mark.asyncio
async def test_self_verifier_bad_json_degrades() -> None:
    verifier = SelfVerifier(MockProvider(response="not json"), depth=1)
    _, steps = await verifier.verify_all([_claim()])
    # Should still return depth+1 steps, all degraded to uncertain
    assert len(steps) == 2
    assert all(s.verdict == "uncertain" for s in steps)


@pytest.mark.asyncio
async def test_self_verifier_invalid_verdict_defaults_to_uncertain() -> None:
    bad = json.dumps({"verdict": "maybe", "reasoning": "hmm", "confidence_delta": 0.0})
    verifier = SelfVerifier(MockProvider(response=bad), depth=0)
    _, steps = await verifier.verify_all([_claim()])
    assert steps[0].verdict == "uncertain"


@pytest.mark.asyncio
async def test_self_verifier_parallel_claims() -> None:
    verifier = SelfVerifier(MockProvider(response=_verify_response()), depth=1)
    claims = [_claim(f"Claim {i}.") for i in range(5)]
    _, steps = await verifier.verify_all(claims)
    # 5 claims × 2 steps each (depth 0 + depth 1)
    assert len(steps) == 10


# ===========================================================================
# CrossChecker
# ===========================================================================

def _cross_response(verdict: str = "supported") -> str:
    return json.dumps({"verdict": verdict, "reasoning": "Second opinion.", "confidence_delta": 0.15})


@pytest.mark.asyncio
async def test_cross_checker_returns_steps_with_depth_minus_one() -> None:
    checker = CrossChecker(MockProvider(response=_cross_response()))
    claims, steps = await checker.check_all([_claim()])
    assert len(steps) == 1
    assert steps[0].depth == -1


@pytest.mark.asyncio
async def test_cross_checker_empty_claims() -> None:
    checker = CrossChecker(MockProvider())
    claims, steps = await checker.check_all([])
    assert claims == []
    assert steps == []


@pytest.mark.asyncio
async def test_cross_checker_nudges_claim_confidence() -> None:
    response = json.dumps({"verdict": "supported", "reasoning": "yes", "confidence_delta": 1.0})
    checker = CrossChecker(MockProvider(response=response))
    base_claim = _claim()
    assert base_claim.confidence == 0.0
    updated_claims, _ = await checker.check_all([base_claim])
    # delta=1.0, weight=0.2 → confidence should increase
    assert updated_claims[0].confidence > 0.0


@pytest.mark.asyncio
async def test_cross_checker_degrades_on_bad_json() -> None:
    checker = CrossChecker(MockProvider(response="not json"))
    claims, steps = await checker.check_all([_claim()])
    assert len(steps) == 1
    assert steps[0].verdict == "uncertain"


# ===========================================================================
# ConfidenceAggregator
# ===========================================================================

def _steps(claim_text: str, verdicts: list[str]) -> list[VerificationStep]:
    return [
        VerificationStep(
            depth=i,
            claim_text=claim_text,
            verdict=v,  # type: ignore[arg-type]
            reasoning="",
            confidence_delta=0.1,
        )
        for i, v in enumerate(verdicts)
    ]


def test_aggregator_stable_supported_high_confidence() -> None:
    agg = ConfidenceAggregator(confidence_threshold=0.4)
    claim = _claim("Paris is in France.")
    steps = _steps("Paris is in France.", ["supported", "supported", "supported"])
    updated, overall, vss = agg.aggregate([claim], steps)
    assert updated[0].confidence > 0.5
    assert updated[0].vss_score == 1.0
    assert updated[0].flip_count == 0
    assert not updated[0].flagged


def test_aggregator_flipping_verdicts_low_vss() -> None:
    agg = ConfidenceAggregator(confidence_threshold=0.5)
    claim = _claim("Claim X.")
    steps = _steps("Claim X.", ["supported", "contradicted", "supported"])
    updated, _, vss = agg.aggregate([claim], steps)
    assert updated[0].flip_count == 2
    assert updated[0].vss_score < 1.0


def test_aggregator_contradicted_flags_claim() -> None:
    agg = ConfidenceAggregator(confidence_threshold=0.5)
    claim = _claim("The sky is green.")
    steps = _steps("The sky is green.", ["contradicted", "contradicted"])
    updated, _, _ = agg.aggregate([claim], steps)
    assert updated[0].flagged is True


def test_aggregator_no_steps_flags_claim() -> None:
    agg = ConfidenceAggregator(confidence_threshold=0.5)
    claim = _claim()
    updated, _, _ = agg.aggregate([claim], [])
    assert updated[0].flagged is True


def test_aggregator_overall_confidence_average() -> None:
    agg = ConfidenceAggregator(confidence_threshold=0.1)
    claims = [_claim("A."), _claim("B.")]
    steps = _steps("A.", ["supported", "supported"]) + _steps("B.", ["supported", "supported"])
    _, overall, _ = agg.aggregate(claims, steps)
    assert 0.0 <= overall <= 1.0


def test_aggregator_vss_average_across_claims() -> None:
    agg = ConfidenceAggregator(confidence_threshold=0.1)
    claims = [_claim("A."), _claim("B.")]
    # A is stable, B flips
    steps = _steps("A.", ["supported", "supported"]) + _steps("B.", ["supported", "contradicted"])
    _, _, vss = agg.aggregate(claims, steps)
    assert 0.0 < vss < 1.0


def test_aggregator_cross_check_included() -> None:
    agg = ConfidenceAggregator(confidence_threshold=0.5)
    claim = _claim("Paris is in France.")
    self_steps = _steps("Paris is in France.", ["supported", "supported"])
    cross = [
        VerificationStep(
            depth=-1,
            claim_text="Paris is in France.",
            verdict="supported",
            reasoning="confirmed",
            confidence_delta=0.5,
        )
    ]
    updated, _, _ = agg.aggregate([claim], self_steps, cross_check_steps=cross)
    # Should not flip — cross agrees
    assert updated[0].confidence > 0.5
