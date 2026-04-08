"""Unit tests for Varity data models (Pydantic v2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from varity.models import Claim, CheckResult, VarityConfig, VerificationStep


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------

def test_claim_defaults() -> None:
    c = Claim(text="Paris is in France.", claim_type="factual", source_span=(0, 20))
    assert c.confidence == 0.0
    assert c.flagged is False
    assert c.vss_score == 1.0
    assert c.flip_count == 0


def test_claim_invalid_type_raises() -> None:
    with pytest.raises(ValidationError):
        Claim(text="x", claim_type="unknown", source_span=(0, 1))  # type: ignore[arg-type]


def test_claim_vss_score_clamped() -> None:
    with pytest.raises(ValidationError):
        Claim(text="x", claim_type="factual", source_span=(0, 1), vss_score=1.5)


# ---------------------------------------------------------------------------
# VerificationStep
# ---------------------------------------------------------------------------

def test_verification_step_valid() -> None:
    step = VerificationStep(
        depth=0,
        claim_text="x",
        verdict="supported",
        reasoning="Looks good.",
        confidence_delta=0.2,
    )
    assert step.verdict == "supported"


def test_verification_step_invalid_verdict() -> None:
    with pytest.raises(ValidationError):
        VerificationStep(
            depth=0,
            claim_text="x",
            verdict="maybe",  # type: ignore[arg-type]
            reasoning="",
        )


def test_verification_step_delta_clamped() -> None:
    with pytest.raises(ValidationError):
        VerificationStep(
            depth=0, claim_text="x", verdict="supported", reasoning="", confidence_delta=2.0
        )


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------

def test_check_result_defaults() -> None:
    result = CheckResult(
        original_response="Hello",
        claims=[],
        flagged_claims=[],
        overall_confidence=0.8,
        vss_score=0.9,
        verification_chain=[],
        duration_ms=100,
    )
    assert result.corrected_response is None
    assert result.token_usage == {}


def test_check_result_confidence_clamped() -> None:
    with pytest.raises(ValidationError):
        CheckResult(
            original_response="x",
            claims=[],
            flagged_claims=[],
            overall_confidence=1.5,
            vss_score=1.0,
            verification_chain=[],
            duration_ms=0,
        )


# ---------------------------------------------------------------------------
# VarityConfig
# ---------------------------------------------------------------------------

def test_varity_config_defaults() -> None:
    cfg = VarityConfig()
    assert cfg.depth == 2
    assert cfg.strategy == "full"
    assert cfg.confidence_threshold == 0.5


def test_varity_config_effective_depth_quick() -> None:
    cfg = VarityConfig(strategy="quick", depth=4)
    assert cfg.effective_depth == 1


def test_varity_config_effective_depth_paranoid() -> None:
    cfg = VarityConfig(strategy="paranoid", depth=2)
    assert cfg.effective_depth == 4


def test_varity_config_effective_depth_full() -> None:
    cfg = VarityConfig(strategy="full", depth=3)
    assert cfg.effective_depth == 3


def test_varity_config_invalid_strategy() -> None:
    with pytest.raises(ValidationError):
        VarityConfig(strategy="turbo")  # type: ignore[arg-type]


def test_varity_config_depth_bounds() -> None:
    with pytest.raises(ValidationError):
        VarityConfig(depth=10)
