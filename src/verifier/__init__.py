"""Claim-level verifier module for detecting hallucinations via NLI entailment."""

from src.verifier.claim_verifier import ClaimVerifier, LocalNLIEntailmentScorer

__all__ = ["ClaimVerifier", "LocalNLIEntailmentScorer"]
