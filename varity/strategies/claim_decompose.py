"""Claim decomposition strategy: response text → list[Claim]."""

from __future__ import annotations

import logging
from typing import Any

from varity.exceptions import DecompositionError
from varity.models import Claim
from varity.prompts import DECOMPOSE_SYSTEM, DECOMPOSE_USER
from varity.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_VALID_TYPES = {"factual", "temporal", "numerical", "causal", "opinion"}


class ClaimDecomposer:
    """Decomposes an LLM response into a list of atomic :class:`~varity.models.Claim` objects.

    Uses the provider to call an LLM with a structured decomposition prompt and
    parses the JSON response into typed Claim instances.  Degrades gracefully:
    if the provider call fails, returns an empty list rather than raising.
    """

    def __init__(self, provider: BaseLLMProvider, max_claims: int = 20) -> None:
        """Initialise the decomposer.

        Args:
            provider: LLM provider used for decomposition.
            max_claims: Upper limit on the number of claims to extract.
        """
        self._provider = provider
        self._max_claims = max_claims

    async def decompose(self, response: str) -> list[Claim]:
        """Decompose *response* into atomic claims.

        Args:
            response: The LLM response text to decompose.

        Returns:
            List of :class:`~varity.models.Claim` instances.  Empty list on
            failure (graceful degradation).
        """
        if not response.strip():
            return []

        prompt = DECOMPOSE_USER.format(
            max_claims=self._max_claims,
            response=response,
        )
        try:
            data = await self._provider.complete_json(prompt, system=DECOMPOSE_SYSTEM)
            return self._parse(data, response)
        except DecompositionError as exc:
            logger.warning("ClaimDecomposer: JSON parse failed — %s", exc)
            return []
        except Exception as exc:
            logger.warning("ClaimDecomposer: provider call failed — %s", exc)
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse(self, data: dict[str, Any], original: str) -> list[Claim]:
        """Parse the raw JSON dict into a list of Claim objects.

        Args:
            data: Parsed JSON from the LLM.
            original: Original response text (used to validate source_span).

        Returns:
            List of valid :class:`~varity.models.Claim` instances.

        Raises:
            DecompositionError: If the JSON structure is missing required keys.
        """
        raw_claims = data.get("claims")
        if not isinstance(raw_claims, list):
            raise DecompositionError(
                "Decomposition response missing 'claims' list."
            )

        claims: list[Claim] = []
        for item in raw_claims[: self._max_claims]:
            claim = self._parse_one(item, original)
            if claim is not None:
                claims.append(claim)
        return claims

    def _parse_one(self, item: Any, original: str) -> Claim | None:
        """Parse a single claim dict, returning None if it is malformed.

        Args:
            item: A single entry from the 'claims' list.
            original: Original response text.

        Returns:
            :class:`~varity.models.Claim` or ``None`` if malformed.
        """
        if not isinstance(item, dict):
            return None

        text = item.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return None

        claim_type = item.get("claim_type", "factual")
        if claim_type not in _VALID_TYPES:
            claim_type = "factual"

        raw_span = item.get("source_span", [0, len(original)])
        if (
            isinstance(raw_span, (list, tuple))
            and len(raw_span) == 2
            and all(isinstance(v, int) for v in raw_span)
        ):
            span: tuple[int, int] = (int(raw_span[0]), int(raw_span[1]))
        else:
            span = (0, len(original))

        return Claim(
            text=text.strip(),
            claim_type=claim_type,
            source_span=span,
        )
