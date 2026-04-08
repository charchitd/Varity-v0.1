"""Cross-check strategy: independent second-opinion verification."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, cast

from varity.models import Claim, VerificationStep
from varity.prompts import CROSS_CHECK_SYSTEM, CROSS_CHECK_USER
from varity.providers.base import BaseLLMProvider

_Verdict = Literal["supported", "contradicted", "uncertain"]

logger = logging.getLogger(__name__)

_VALID_VERDICTS = {"supported", "contradicted", "uncertain"}


class CrossChecker:
    """Provides an independent second opinion on each claim.

    Uses a different prompt framing than :class:`~varity.strategies.self_verify.SelfVerifier`
    (no depth loop, no prior verdicts) to avoid anchoring bias.  The cross-check
    result is stored as an extra :class:`~varity.models.VerificationStep` at
    ``depth = -1`` and also applied as a confidence nudge on the claim.

    All claims are checked in parallel via :func:`asyncio.gather`.
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        """Initialise the cross-checker.

        Args:
            provider: LLM provider used for cross-check calls.
        """
        self._provider = provider

    async def check_all(
        self, claims: list[Claim]
    ) -> tuple[list[Claim], list[VerificationStep]]:
        """Cross-check all claims in parallel.

        Args:
            claims: Claims to cross-check.

        Returns:
            Tuple of (updated claims, list of cross-check VerificationSteps).
            Each step has ``depth = -1`` to distinguish it from self-verify steps.
        """
        if not claims:
            return [], []

        results = await asyncio.gather(
            *(self._check_one(c) for c in claims),
            return_exceptions=True,
        )

        updated_claims: list[Claim] = []
        cross_steps: list[VerificationStep] = []

        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning("CrossChecker: claim %d failed — %s", i, result)
                updated_claims.append(claims[i])
                continue
            claim, step = result
            updated_claims.append(claim)
            cross_steps.append(step)

        return updated_claims, cross_steps

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _check_one(self, claim: Claim) -> tuple[Claim, VerificationStep]:
        """Run a single cross-check on one claim.

        Applies the cross-check confidence_delta directly to the claim's
        existing confidence score.

        Args:
            claim: The claim to cross-check.

        Returns:
            Tuple of (updated claim, VerificationStep with depth=-1).
        """
        prompt = CROSS_CHECK_USER.format(claim_text=claim.text)
        try:
            data = await self._provider.complete_json(prompt, system=CROSS_CHECK_SYSTEM)
            raw_verdict = str(data.get("verdict", "uncertain")).lower()
            verdict: _Verdict = cast(
                _Verdict, raw_verdict if raw_verdict in _VALID_VERDICTS else "uncertain"
            )

            raw_delta = data.get("confidence_delta", 0.0)
            try:
                delta = float(raw_delta)
                delta = max(-1.0, min(1.0, delta))
            except (TypeError, ValueError):
                delta = 0.0

            step = VerificationStep(
                depth=-1,
                claim_text=claim.text,
                verdict=verdict,
                reasoning=str(data.get("reasoning", "")),
                confidence_delta=delta,
            )
            # Nudge claim confidence by a fraction of the cross-check delta
            updated = claim.model_copy(
                update={
                    "confidence": max(0.0, min(1.0, claim.confidence + delta * 0.2))
                }
            )
            return updated, step

        except Exception as exc:
            logger.warning("CrossChecker: provider failed for '%s' — %s", claim.text[:40], exc)
            fallback_step = VerificationStep(
                depth=-1,
                claim_text=claim.text,
                verdict="uncertain",
                reasoning="Cross-check failed — degraded to uncertain.",
                confidence_delta=0.0,
            )
            return claim, fallback_step
