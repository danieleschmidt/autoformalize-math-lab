"""Proof verification and feedback interfaces.

This module provides interfaces to proof assistants for verifying generated
formal proofs and extracting error feedback for self-correction.

Modules:
    base: Base verifier interface
    lean_verifier: Lean 4 proof verification
    isabelle_verifier: Isabelle/HOL proof verification
    coq_verifier: Coq proof verification
    error_parser: Error message parsing and analysis
    feedback_extractor: Structured feedback extraction
"""

__all__ = [
    "BaseVerifier",
    "LeanVerifier",
    "IsabelleVerifier",
    "CoqVerifier",
    "ErrorParser",
    "FeedbackExtractor",
]