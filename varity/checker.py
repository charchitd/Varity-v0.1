"""RecursiveChecker: main pipeline orchestrator for Varity."""

from __future__ import annotations

import json
import logging

from varity.exceptions import DecompositionError
from varity.models import CheckResult, Claim, VarityConfig, VerificationStep
from varity.prompts import (
    CORRECT_SYSTEM,
    CORRECT_USER,
    CROSS_CHECK_SYSTEM,
    CROSS_CHECK_USER,
    DECOMPOSE_SYSTEM,
    DECOMPOSE_USER,
    VERIFY_SYSTEM,
    VERIFY_USER,
)
from varity.providers.base import BaseLLMProvider
from varity.strategies.claim_decompose import ClaimDecomposer
from varity.strategies.confidence import ConfidenceAggregator
from varity.strategies.cross_check import CrossChecker
from varity.strategies.self_verify import SelfVerifier
from varity.utils import count_tokens, now_ms

logger = logging.getLogger(__name__)


class RecursiveChecker:
    """Orchestrates the full five-stage Varity verification pipeline.

    Stages (run in order):
    1. :class:`~varity.strategies.claim_decompose.ClaimDecomposer`
    2. :class:`~varity.strategies.self_verify.SelfVerifier`
    3. :class:`~varity.strategies.cross_check.CrossChecker`
    4. :class:`~varity.strategies.confidence.ConfidenceAggregator`
    5. Correction generator (inline, uses the provider)
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        config: VarityConfig | None = None,
    ) -> None:
        """Initialise the checker with a provider and optional config.

        Args:
            provider: LLM provider used for all pipeline stages.
            config: Optional :class:`~varity.models.VarityConfig`; defaults
                are used when omitted.
        """
        self._provider = provider
        self._config = config or VarityConfig()
        self._decomposer = ClaimDecomposer(
            provider=provider,
            max_claims=self._config.max_claims,
        )
        self._verifier = SelfVerifier(
            provider=provider,
            depth=self._config.effective_depth,
        )
        self._cross_checker = CrossChecker(provider=provider)
        self._aggregator = ConfidenceAggregator(
            confidence_threshold=self._config.confidence_threshold,
        )

    async def run(self, response: str) -> CheckResult:
        """Execute the full verification pipeline on *response*.

        Graceful degradation: each stage that fails is logged and skipped;
        the pipeline always returns a :class:`~varity.models.CheckResult`.

        Args:
            response: The LLM response text to verify.

        Returns:
            :class:`~varity.models.CheckResult` with scores and optional
            corrected response.
        """
        start_ms = now_ms()

        # ------------------------------------------------------------------
        # Stage 1 — Decompose
        # ------------------------------------------------------------------
        try:
            claims = await self._decomposer.decompose(response)
        except DecompositionError as exc:
            logger.error("Stage 1 (decompose) failed: %s", exc)
            claims = []

        logger.debug("Stage 1 complete: %d claims", len(claims))

        if not claims:
            return self._empty_result(response, now_ms() - start_ms)

        # ------------------------------------------------------------------
        # Stage 2 — Self-verify (parallel per claim)
        # ------------------------------------------------------------------
        try:
            claims, verification_chain = await self._verifier.verify_all(claims)
        except Exception as exc:
            logger.error("Stage 2 (self-verify) failed: %s", exc)
            verification_chain = []

        logger.debug("Stage 2 complete: %d verification steps", len(verification_chain))

        # ------------------------------------------------------------------
        # Stage 3 — Cross-check (parallel per claim)
        # ------------------------------------------------------------------
        try:
            claims, cross_steps = await self._cross_checker.check_all(claims)
        except Exception as exc:
            logger.error("Stage 3 (cross-check) failed: %s", exc)
            cross_steps = []

        logger.debug("Stage 3 complete: %d cross-check steps", len(cross_steps))

        # ------------------------------------------------------------------
        # Stage 4 — Aggregate confidence + VSS
        # ------------------------------------------------------------------
        try:
            claims, overall_confidence, vss_score = self._aggregator.aggregate(
                claims=claims,
                verification_chain=verification_chain,
                cross_check_steps=cross_steps,
            )
        except Exception as exc:
            logger.error("Stage 4 (confidence) failed: %s", exc)
            overall_confidence = 0.5
            vss_score = 0.5

        flagged = [c for c in claims if c.flagged]
        logger.debug(
            "Stage 4 complete: overall_confidence=%.3f vss=%.3f flagged=%d",
            overall_confidence,
            vss_score,
            len(flagged),
        )

        # ------------------------------------------------------------------
        # Stage 5 — Correction generator
        # ------------------------------------------------------------------
        corrected_response: str | None = None
        if flagged:
            try:
                corrected_response = await self._generate_correction(response, flagged)
            except Exception as exc:
                logger.warning("Stage 5 (correction) failed: %s", exc)

        all_steps = list(verification_chain) + list(cross_steps)
        token_usage = self._estimate_token_usage(
            response=response,
            claims=claims,
            verification_chain=all_steps,
            corrected_response=corrected_response,
        )

        return CheckResult(
            original_response=response,
            claims=claims,
            flagged_claims=flagged,
            overall_confidence=overall_confidence,
            vss_score=vss_score,
            corrected_response=corrected_response,
            verification_chain=all_steps,
            token_usage=token_usage,
            duration_ms=now_ms() - start_ms,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _generate_correction(
        self, original_response: str, flagged_claims: list[Claim]
    ) -> str:
        """Ask the LLM to rewrite *original_response* correcting flagged claims.

        Args:
            original_response: The original LLM response text.
            flagged_claims: Claims that were flagged as uncertain/contradicted.

        Returns:
            Corrected response string.

        Raises:
            Exception: Propagated from the provider on network failure.
        """
        flagged_str = "\n".join(
            f"- [{c.claim_type}] {c.text} (confidence={c.confidence:.2f}, vss={c.vss_score:.2f})"
            for c in flagged_claims
        )
        prompt = CORRECT_USER.format(
            original_response=original_response,
            flagged_claims=flagged_str,
        )
        return await self._provider.complete(prompt, system=CORRECT_SYSTEM)

    def _estimate_token_usage(
        self,
        response: str,
        claims: list[Claim],
        verification_chain: list[VerificationStep],
        corrected_response: str | None,
    ) -> dict[str, int]:
        """Estimate token usage across all pipeline stages.

        Uses tiktoken to count tokens from reconstructed prompts and known
        completion text.  This is an estimate — actual billed tokens may differ
        slightly depending on provider-side formatting.

        Args:
            response: Original LLM response text.
            claims: Decomposed claims.
            verification_chain: All VerificationStep objects (self-verify + cross-check).
            corrected_response: Generated correction, or None.

        Returns:
            Dict with ``prompt_tokens``, ``completion_tokens``, ``total_tokens``.
        """
        model = self._provider.model
        prompt_tokens = 0
        completion_tokens = 0

        # Stage 1 — Decompose
        decompose_prompt = DECOMPOSE_USER.format(
            max_claims=self._config.max_claims, response=response
        )
        prompt_tokens += count_tokens(DECOMPOSE_SYSTEM + decompose_prompt, model)
        claims_json = json.dumps(
            [{"text": c.text, "claim_type": c.claim_type} for c in claims]
        )
        completion_tokens += count_tokens(claims_json, model)

        # Stage 2 — Self-verify (depth >= 0)
        for step in verification_chain:
            if step.depth >= 0:
                verify_prompt = VERIFY_USER.format(
                    depth=step.depth,
                    claim_text=step.claim_text,
                    previous_verdicts="",
                )
                prompt_tokens += count_tokens(VERIFY_SYSTEM + verify_prompt, model)
                completion_tokens += count_tokens(step.reasoning, model)

        # Stage 3 — Cross-check (depth == -1)
        for step in verification_chain:
            if step.depth == -1:
                cross_prompt = CROSS_CHECK_USER.format(claim_text=step.claim_text)
                prompt_tokens += count_tokens(CROSS_CHECK_SYSTEM + cross_prompt, model)
                completion_tokens += count_tokens(step.reasoning, model)

        # Stage 5 — Correction
        if corrected_response:
            correct_prompt = CORRECT_USER.format(
                original_response=response,
                flagged_claims="",
            )
            prompt_tokens += count_tokens(CORRECT_SYSTEM + correct_prompt, model)
            completion_tokens += count_tokens(corrected_response, model)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    @staticmethod
    def _empty_result(response: str, duration_ms: int) -> CheckResult:
        """Return an empty CheckResult when decomposition produced no claims.

        Args:
            response: The original response text.
            duration_ms: Elapsed time in milliseconds.

        Returns:
            :class:`~varity.models.CheckResult` with zero claims.
        """
        return CheckResult(
            original_response=response,
            claims=[],
            flagged_claims=[],
            overall_confidence=1.0,
            vss_score=1.0,
            corrected_response=None,
            verification_chain=[],
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            duration_ms=duration_ms,
        )
