"""Pydantic v2 data models for Varity."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A single atomic claim extracted from an LLM response."""

    text: str
    claim_type: Literal["factual", "temporal", "numerical", "causal", "opinion"]
    source_span: tuple[int, int]
    confidence: float = 0.0
    flagged: bool = False
    verification_notes: str = ""
    vss_score: float = Field(default=1.0, ge=0.0, le=1.0)
    flip_count: int = 0


class VerificationStep(BaseModel):
    """A single step in the recursive verification chain."""

    depth: int
    claim_text: str
    verdict: Literal["supported", "contradicted", "uncertain"]
    reasoning: str
    confidence_delta: float = Field(default=0.0, ge=-1.0, le=1.0)


class CheckResult(BaseModel):
    """The full output of a Varity check run."""

    original_response: str
    claims: list[Claim]
    flagged_claims: list[Claim]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    vss_score: float = Field(ge=0.0, le=1.0)
    corrected_response: Optional[str] = None
    verification_chain: list[VerificationStep]
    token_usage: dict[str, int] = Field(default_factory=dict)
    duration_ms: int


class VarityConfig(BaseModel):
    """Configuration for a Varity check run."""

    depth: int = Field(default=2, ge=0, le=5)
    strategy: Literal["quick", "full", "paranoid"] = "full"
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_claims: int = Field(default=20, ge=1, le=100)
    timeout: int = Field(default=30, ge=5, le=300)

    @property
    def effective_depth(self) -> int:
        """Map strategy to verification depth."""
        overrides = {"quick": 1, "full": self.depth, "paranoid": max(self.depth, 4)}
        return overrides[self.strategy]
