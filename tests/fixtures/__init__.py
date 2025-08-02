"""Test fixtures for autoformalize-math-lab."""

from .sample_theorems import (
    ALGEBRA_THEOREMS,
    NUMBER_THEORY_THEOREMS,
    ANALYSIS_THEOREMS,
    ALL_THEOREMS,
    get_theorem_by_domain,
    get_theorem_by_difficulty,
    get_sample_latex_errors,
    get_sample_proof_assistant_errors,
)

__all__ = [
    "ALGEBRA_THEOREMS",
    "NUMBER_THEORY_THEOREMS", 
    "ANALYSIS_THEOREMS",
    "ALL_THEOREMS",
    "get_theorem_by_domain",
    "get_theorem_by_difficulty",
    "get_sample_latex_errors",
    "get_sample_proof_assistant_errors",
]