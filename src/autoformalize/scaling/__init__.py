"""High-performance scaling and optimization components.

This module provides advanced scaling capabilities for mathematical
formalization pipelines including distributed processing, auto-scaling,
and performance optimization.
"""

from .distributed_pipeline import DistributedFormalizationPipeline
from .auto_scaler import AutoScaler
from .load_balancer import LoadBalancer
from .performance_optimizer import PerformanceOptimizer
from .resource_manager import ResourceManager

__all__ = [
    "DistributedFormalizationPipeline",
    "AutoScaler", 
    "LoadBalancer",
    "PerformanceOptimizer",
    "ResourceManager"
]