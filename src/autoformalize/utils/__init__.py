"""Utility functions and common tools.

This module provides utility functions, data structures, and helper tools
used throughout the autoformalize package.

Modules:
    logging_config: Structured logging configuration
    file_utils: File and path manipulation utilities
    math_utils: Mathematical notation and processing utilities
    caching: Caching mechanisms for expensive operations
    config_loader: Configuration file loading and validation
    time_utils: Timing and performance measurement utilities
"""

try:
    from .logging_config import setup_logger
    from .metrics import FormalizationMetrics
    from .caching import CacheManager, CacheStrategy
    from .concurrency import ResourcePool, AsyncBatch
    from .resilience import retry_async, CircuitBreaker
except ImportError:
    # Handle missing dependencies gracefully
    pass

__all__ = [
    "setup_logger",
    "FormalizationMetrics",
    "CacheManager",
    "CacheStrategy", 
    "ResourcePool",
    "AsyncBatch",
    "retry_async",
    "CircuitBreaker",
]