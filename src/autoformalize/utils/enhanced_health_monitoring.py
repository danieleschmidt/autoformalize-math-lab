"""Advanced health monitoring and observability system.

This module provides comprehensive health monitoring, metrics collection,
and observability features for the formalization pipeline.
"""

import asyncio
import time
import logging
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # Mock psutil for basic functionality
    class psutil:
        @staticmethod
        def cpu_percent(interval=1):
            return 25.0  # Mock CPU usage
        
        @staticmethod
        def virtual_memory():
            class Memory:
                percent = 45.0
            return Memory()
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
from pathlib import Path

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

from .logging_config import setup_logger


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    
    @property
    def status(self) -> HealthStatus:
        """Get status based on thresholds."""
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


class AdvancedHealthMonitor:
    """Advanced health monitoring system with real-time metrics."""
    
    def __init__(
        self,
        enable_prometheus: bool = True,
        prometheus_port: int = 8000,
        collection_interval: float = 30.0
    ):
        """Initialize health monitoring system."""
        self.enable_prometheus = enable_prometheus and HAS_PROMETHEUS
        self.collection_interval = collection_interval
        
        self.logger = setup_logger(__name__)
        
        # Metrics storage
        self.current_metrics: Dict[str, HealthMetric] = {}
        
        # Performance counters
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        
        # Health thresholds
        self.thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "error_rate": {"warning": 5.0, "critical": 15.0}
        }
        
        # Prometheus metrics
        if self.enable_prometheus:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics."""
        if not HAS_PROMETHEUS:
            return
        
        self.prometheus_metrics = {
            "system_cpu_usage": Gauge("system_cpu_usage_percent", "System CPU usage percentage"),
            "system_memory_usage": Gauge("system_memory_usage_percent", "System memory usage percentage"),
            "error_rate": Gauge("error_rate_percent", "Error rate percentage")
        }
    
    def record_operation(self, operation_type: str, success: bool) -> None:
        """Record an operation for metrics."""
        self.operation_counts[operation_type] += 1
        if not success:
            self.error_counts[operation_type] += 1
    
    async def get_current_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        # Collect current metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Calculate error rate
        total_operations = sum(self.operation_counts.values())
        total_errors = sum(self.error_counts.values())
        error_rate = (total_errors / total_operations * 100) if total_operations > 0 else 0.0
        
        # Update current metrics
        self.current_metrics["cpu_usage"] = HealthMetric("cpu_usage", cpu_percent, 70.0, 90.0, "%")
        self.current_metrics["memory_usage"] = HealthMetric("memory_usage", memory.percent, 80.0, 95.0, "%")
        self.current_metrics["error_rate"] = HealthMetric("error_rate", error_rate, 5.0, 15.0, "%")
        
        # Determine overall status
        statuses = [metric.status for metric in self.current_metrics.values()]
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "overall_status": overall_status.value,
            "metrics": {name: {"value": m.value, "unit": m.unit, "status": m.status.value} 
                       for name, m in self.current_metrics.items()},
            "operation_counts": dict(self.operation_counts),
            "error_counts": dict(self.error_counts)
        }