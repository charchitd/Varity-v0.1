"""Strategy classes for the Varity verification pipeline."""

from varity.strategies.claim_decompose import ClaimDecomposer
from varity.strategies.confidence import ConfidenceAggregator
from varity.strategies.cross_check import CrossChecker
from varity.strategies.self_verify import SelfVerifier

__all__ = [
    "ClaimDecomposer",
    "SelfVerifier",
    "CrossChecker",
    "ConfidenceAggregator",
]
