"""Self-verification strategy: recursive depth-N verdict loop per claim."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, cast

from varity.exceptions import VerificationError
from varity.models import Claim, VerificationStep
from varity.prompts import VERIFY_SYSTEM, VERIFY_USER
from varity.providers.base import BaseLLMProvider

_Verdict = Literal["supported", "contradicted", "uncertain"]

logger = logging.getLogger(__name__)

_VALID_VERDICTS = {"supported", "contradicted", "uncertain"}


class SelfVerifier:
    """Recursively verifies each claim by asking the LLM to re-examine its own verdict.

    For each claim the verifier runs *depth + 1* passes (0 through *depth*).
    Each pass sees the verdicts from all prior passes, forcing the model to
    challenge its own reasoning.  Verdict flips across passes are counted and
    later used to compute the Verdict Stability Score (VSS).

    All claims are verified in parallel via :func:`asyncio.gather`.
    """

    def __init__(self, provider: BaseLLMProvider, depth: int = 2) -> None:
        """Initialise the verifier.

        Args:
            provider: LLM provider used for verification calls.
            depth: Maximum verification depth (0 = single pass, 2 = three passes).
        """
        self._provider = provider
        self._depth = depth

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify_all(
        self, claims: list[Claim]
    ) -> tuple[list[Claim], list[VerificationStep]]:
        """Verify all claims in parallel.

        Args:
            claims: Claims to verify.

        Returns:
            Tuple of (same claims list, flat list of all VerificationStep objects).
        """
        if not claims:
            return [], []

        results = await asyncio.gather(
            *(self._verify_claim(c) for c in claims),
            return_exceptions=True,
        )

        all_steps: list[VerificationStep] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("SelfVerifier: claim verification raised — %s", result)
                continue
            _claim_r, steps = result
            all_steps.extend(steps)

        return claims, all_steps

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _verify_claim(
        self, claim: Claim
    ) -> tuple[Claim, list[VerificationStep]]:
        """Run depth+1 verification passes on a single claim.

        Args:
            claim: The claim to verify.

        Returns:
            Tuple of (claim, list of VerificationStep for depths 0..depth).
        """
        steps: list[VerificationStep] = []
        previous_verdicts: list[str] = []

        for d in range(self._depth + 1):
            step = await self._verify_at_depth(claim, d, previous_verdicts)
            steps.append(step)
            previous_verdicts.append(step.verdict)

        return claim, steps

    async def _verify_at_depth(
        self,
        claim: Claim,
        depth: int,
        previous_verdicts: list[str],
    ) -> VerificationStep:
        """Run one verification pass at the given depth.

        Degrades gracefully: returns an ``uncertain`` step on any failure.

        Args:
            claim: The claim being verified.
            depth: Current verification depth index.
            previous_verdicts: Verdicts from shallower depths (may be empty).

        Returns:
            A :class:`~varity.models.VerificationStep`.
        """
        prior_str = (
            ", ".join(previous_verdicts) if previous_verdicts else "None"
        )
        prompt = VERIFY_USER.format(
            depth=depth,
            claim_text=claim.text,
            previous_verdicts=prior_str,
        )
        try:
            data = await self._provider.complete_json(prompt, system=VERIFY_SYSTEM)
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

            return VerificationStep(
                depth=depth,
                claim_text=claim.text,
                verdict=verdict,
                reasoning=str(data.get("reasoning", "")),
                confidence_delta=delta,
            )
        except VerificationError as exc:
            logger.warning("SelfVerifier depth=%d: JSON parse failed — %s", depth, exc)
        except Exception as exc:
            logger.warning("SelfVerifier depth=%d: provider failed — %s", depth, exc)

        return VerificationStep(
            depth=depth,
            claim_text=claim.text,
            verdict="uncertain",
            reasoning="Verification failed — degraded to uncertain.",
            confidence_delta=0.0,
        )
