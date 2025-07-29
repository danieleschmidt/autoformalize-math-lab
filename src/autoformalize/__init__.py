"""Autoformalize Math Lab - Automated Mathematical Formalization.

This package provides tools for converting LaTeX proofs to formal proof assistant code
(Lean 4, Isabelle/HOL, Coq) using large language models and self-correcting pipelines.

Main Components:
    - core: Core formalization logic and pipeline orchestration
    - parsers: LaTeX and mathematical content parsing
    - generators: Proof assistant code generation
    - verifiers: Proof verification and feedback interfaces
    - models: LLM integration and prompt engineering
    - utils: Utility functions and common tools
    - datasets: Benchmark datasets and evaluation frameworks

Example:
    Basic usage for formalizing a LaTeX proof:

    >>> from autoformalize import FormalizationPipeline
    >>> pipeline = FormalizationPipeline(target_system="lean4")
    >>> latex_proof = '''
    ... \\\\begin{theorem}
    ... For any prime $p > 2$, $p$ is odd.
    ... \\\\end{theorem}
    ... \\\\begin{proof}
    ... Since $p > 2$ and $p$ is prime, $p$ is not divisible by 2.
    ... Therefore $p$ is odd.
    ... \\\\end{proof}
    ... '''
    >>> lean_code = pipeline.formalize(latex_proof)
    >>> print(lean_code)
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

# Import main classes for convenience
try:
    from .core.pipeline import FormalizationPipeline
    from .core.self_correcting import SelfCorrectingPipeline
    from .core.cross_system import CrossSystemTranslator
    from .core.mathlib_aligner import MathlibAligner
except ImportError:
    # Handle case where dependencies are not yet installed
    pass

# Package metadata
__all__ = [
    "FormalizationPipeline",
    "SelfCorrectingPipeline", 
    "CrossSystemTranslator",
    "MathlibAligner",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Version compatibility check
import sys
if sys.version_info < (3, 9):
    raise RuntimeError(
        "autoformalize-math-lab requires Python 3.9 or higher. "
        f"You are using Python {sys.version_info.major}.{sys.version_info.minor}."
    )

# Optional: Register proof assistant plugins
def _register_plugins():
    """Register proof assistant plugins if available."""
    try:
        import pkg_resources
        for entry_point in pkg_resources.iter_entry_points('autoformalize.plugins'):
            try:
                entry_point.load()
            except Exception:
                # Silently ignore plugin loading failures
                pass
    except ImportError:
        # pkg_resources not available
        pass

# Initialize plugins on import
_register_plugins()