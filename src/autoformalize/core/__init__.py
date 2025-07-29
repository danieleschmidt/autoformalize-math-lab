"""Core formalization logic and pipeline orchestration.

This module contains the main formalization pipeline components that coordinate
the conversion of LaTeX mathematical content to formal proof assistant code.

Modules:
    pipeline: Main formalization pipeline implementation
    self_correcting: Self-correcting pipeline with error feedback
    cross_system: Cross-system translation capabilities
    mathlib_aligner: Integration with mathematical libraries
    config: Configuration management
    exceptions: Custom exception classes
"""

__all__ = [
    "FormalizationPipeline",
    "SelfCorrectingPipeline",
    "CrossSystemTranslator", 
    "MathlibAligner",
    "FormalizationConfig",
    "FormalizationError",
]