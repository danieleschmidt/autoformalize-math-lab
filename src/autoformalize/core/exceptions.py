"""Custom exception classes for the formalization pipeline.

This module defines specific exception types that can occur during
the mathematical formalization process.
"""


class FormalizationError(Exception):
    """Base exception for formalization pipeline errors."""
    pass


class ParseError(FormalizationError):
    """Raised when LaTeX parsing fails."""
    pass


class GenerationError(FormalizationError):
    """Raised when formal code generation fails."""
    pass


class VerificationError(FormalizationError):
    """Raised when proof verification fails."""
    pass


class UnsupportedSystemError(FormalizationError):
    """Raised when an unsupported proof assistant system is requested."""
    pass


class ValidationError(FormalizationError):
    """Raised when input validation fails."""
    pass


class ModelError(FormalizationError):
    """Raised when LLM model access fails."""
    pass


class TimeoutError(FormalizationError):
    """Raised when operations exceed timeout limits."""
    pass


class ConfigurationError(FormalizationError):
    """Raised when configuration is invalid or missing."""
    pass
