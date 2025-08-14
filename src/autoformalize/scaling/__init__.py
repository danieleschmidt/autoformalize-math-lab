"""High-performance scaling and optimization components.

This module provides advanced scaling capabilities for mathematical
formalization pipelines including distributed processing, auto-scaling,
and performance optimization.
"""

try:
    from .distributed_coordination import DistributedTaskManager, NodeInfo, NodeRole, ClusterConfig
    __all__ = ["DistributedTaskManager", "NodeInfo", "NodeRole", "ClusterConfig"]
except ImportError:
    __all__ = []