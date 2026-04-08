"""Confidence aggregation and Verdict Stability Score (VSS) computation."""

from __future__ import annotations

import logging

from varity.models import Claim, VerificationStep

logger = logging.getLogger(__name__)

# Baseline confidence by initial verdict
_VERDICT_PRIOR: dict[str, float] = {
    "supported": 0.75,
    "contradicted": 0.15,
    "uncertain": 0.45,
}
# Weight of each subsequent depth update
_DEPTH_UPDATE_WEIGHT = 0.25
# Weight of cross-check delta (depth == -1 steps handled separately)
_CROSS_CHECK_WEIGHT = 0.20


class ConfidenceAggregator:
    """Computes per-claim confidence scores and the Verdict Stability Score (VSS).

    VSS measures how stable a claim's verdict was across recursive verification
    depths.  A claim that flips verdict at every depth gets VSS=0.0; one that
    never flips gets VSS=1.0.

    High flip_count → unstable → hallucination signal.
    Low flip_count  → stable  → likely grounded.
    """

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        """Initialise the aggregator.

        Args:
            confidence_threshold: Claims with confidence below this threshold
                are flagged.
        """
        self._threshold = confidence_threshold

    def aggregate(
        self,
        claims: list[Claim],
        verification_chain: list[VerificationStep],
        cross_check_steps: list[VerificationStep] | None = None,
    ) -> tuple[list[Claim], float, float]:
        """Score all claims and compute overall metrics.

        Args:
            claims: Claims from the decomposition stage.
            verification_chain: All self-verify VerificationStep objects
                (depths 0..N for every claim).
            cross_check_steps: Cross-check steps (depth == -1), optional.

        Returns:
            Tuple of:
            - Updated list of claims (with confidence, vss_score, flip_count, flagged set).
            - overall_confidence (float in [0, 1]).
            - vss_score (float in [0, 1], average across claims).
        """
        # Build lookup: claim_text → list of depth-sorted self-verify steps
        self_steps: dict[str, list[VerificationStep]] = {}
        for step in verification_chain:
            if step.depth >= 0:
                self_steps.setdefault(step.claim_text, []).append(step)
        for steps in self_steps.values():
            steps.sort(key=lambda s: s.depth)

        # Build lookup: claim_text → cross-check step
        cross_map: dict[str, VerificationStep] = {}
        for step in (cross_check_steps or []):
            cross_map[step.claim_text] = step

        updated: list[Claim] = []
        for claim in claims:
            steps = self_steps.get(claim.text, [])
            cross = cross_map.get(claim.text)
            scored = self._score_claim(claim, steps, cross)
            updated.append(scored)

        if not updated:
            return [], 0.0, 1.0

        overall_conf = sum(c.confidence for c in updated) / len(updated)
        overall_vss = sum(c.vss_score for c in updated) / len(updated)

        return updated, round(overall_conf, 4), round(overall_vss, 4)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_claim(
        self,
        claim: Claim,
        steps: list[VerificationStep],
        cross: VerificationStep | None,
    ) -> Claim:
        """Compute confidence and VSS for a single claim.

        Args:
            claim: The claim to score.
            steps: Self-verify steps at depths 0..N (sorted).
            cross: Optional cross-check step (depth=-1).

        Returns:
            Claim with updated confidence, vss_score, flip_count, and flagged.
        """
        if not steps:
            # No verification data — mark uncertain, low confidence
            return claim.model_copy(
                update={
                    "confidence": 0.35,
                    "vss_score": 0.5,
                    "flip_count": 0,
                    "flagged": True,
                    "verification_notes": "No verification data available.",
                }
            )

        # Bayesian-style confidence update
        confidence = _VERDICT_PRIOR[steps[0].verdict]

        prev_verdict = steps[0].verdict
        flip_count = 0

        for step in steps[1:]:
            if step.verdict != prev_verdict:
                flip_count += 1
            prev_verdict = step.verdict
            confidence = _clamp(
                confidence + step.confidence_delta * _DEPTH_UPDATE_WEIGHT
            )

        # Apply cross-check nudge
        if cross is not None:
            if cross.verdict != prev_verdict:
                flip_count += 1
            confidence = _clamp(
                confidence + cross.confidence_delta * _CROSS_CHECK_WEIGHT
            )

        # VSS: 1 - (flips / max_possible_flips)
        max_flips = len(steps) - 1 + (1 if cross else 0)
        vss = 1.0 - (flip_count / max_flips) if max_flips > 0 else 1.0

        # Penalise confidence proportionally to instability
        confidence = _clamp(confidence * (0.5 + 0.5 * vss))

        flagged = confidence < self._threshold

        notes_parts = [f"depth={len(steps) - 1}", f"flips={flip_count}", f"vss={vss:.2f}"]
        if cross:
            notes_parts.append(f"cross={cross.verdict}")

        return claim.model_copy(
            update={
                "confidence": round(confidence, 4),
                "vss_score": round(vss, 4),
                "flip_count": flip_count,
                "flagged": flagged,
                "verification_notes": ", ".join(notes_parts),
            }
        )


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
