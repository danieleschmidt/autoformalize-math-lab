"""Metrics collection and performance tracking.

This module provides comprehensive metrics collection for the formalization
pipeline, including success rates, processing times, and error tracking.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import threading
from enum import Enum

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    Counter = Histogram = Gauge = Summary = CollectorRegistry = None

from .logging_config import setup_logger


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricEvent:
    """Represents a single metric event."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingMetrics:
    """Metrics for a single processing operation."""
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    target_system: Optional[str] = None
    content_length: int = 0
    output_length: int = 0
    correction_rounds: int = 0
    verification_success: Optional[bool] = None
    
    @property
    def processing_time(self) -> float:
        """Get processing time in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "processing_time": self.processing_time,
            "success": self.success,
            "error_message": self.error_message,
            "target_system": self.target_system,
            "content_length": self.content_length,
            "output_length": self.output_length,
            "correction_rounds": self.correction_rounds,
            "verification_success": self.verification_success,
        }


class FormalizationMetrics:
    """Comprehensive metrics collection for the formalization pipeline.
    
    This class tracks various metrics related to the formalization process
    including success rates, processing times, error rates, and system-specific
    performance indicators.
    """
    
    def __init__(self, enable_prometheus: bool = True, max_history: int = 10000):
        """Initialize metrics collection.
        
        Args:
            enable_prometheus: Whether to enable Prometheus metrics
            max_history: Maximum number of historical events to keep
        """
        self.enable_prometheus = enable_prometheus and HAS_PROMETHEUS
        self.max_history = max_history
        self.logger = setup_logger(__name__)
        
        # Thread-safe data structures
        self._lock = threading.RLock()
        
        # Historical data
        self.events: deque = deque(maxlen=max_history)
        self.processing_history: deque = deque(maxlen=max_history)
        
        # Aggregated statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "by_system": defaultdict(lambda: {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "avg_processing_time": 0.0,
                "total_processing_time": 0.0
            }),
            "error_counts": defaultdict(int),
            "verification_stats": {
                "attempted": 0,
                "successful": 0,
                "failed": 0
            }
        }
        
        # Setup Prometheus metrics if available
        if self.enable_prometheus:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics collectors."""
        self.registry = CollectorRegistry()
        
        # Counters
        self.requests_total = Counter(
            'autoformalize_requests_total',
            'Total number of formalization requests',
            ['target_system', 'status'],
            registry=self.registry
        )
        
        self.processing_time_seconds = Histogram(
            'autoformalize_processing_time_seconds',
            'Time spent processing formalization requests',
            ['target_system'],
            registry=self.registry
        )
        
        self.verification_attempts = Counter(
            'autoformalize_verification_attempts_total',
            'Total number of verification attempts',
            ['target_system', 'result'],
            registry=self.registry
        )
        
        self.content_length_bytes = Histogram(
            'autoformalize_content_length_bytes',
            'Length of input content in bytes',
            ['target_system'],
            registry=self.registry
        )
        
        self.output_length_bytes = Histogram(
            'autoformalize_output_length_bytes',
            'Length of generated output in bytes',
            ['target_system'],
            registry=self.registry
        )
        
        self.correction_rounds = Histogram(
            'autoformalize_correction_rounds',
            'Number of correction rounds needed',
            ['target_system'],
            registry=self.registry
        )
        
        # Gauges
        self.active_requests = Gauge(
            'autoformalize_active_requests',
            'Number of currently active requests',
            registry=self.registry
        )
        
        self.success_rate = Gauge(
            'autoformalize_success_rate',
            'Current success rate (0-1)',
            ['target_system'],
            registry=self.registry
        )
    
    def start_processing(self, target_system: Optional[str] = None, content_length: int = 0) -> ProcessingMetrics:
        """Start tracking a new processing operation.
        
        Args:
            target_system: Target proof assistant system
            content_length: Length of input content
            
        Returns:
            ProcessingMetrics object to track the operation
        """
        metrics = ProcessingMetrics(
            start_time=time.time(),
            target_system=target_system,
            content_length=content_length
        )
        
        with self._lock:
            if self.enable_prometheus:
                self.active_requests.inc()
                if target_system:
                    self.content_length_bytes.labels(target_system=target_system).observe(content_length)
        
        return metrics
    
    def record_formalization(
        self,
        success: bool,
        target_system: Optional[str] = None,
        processing_time: Optional[float] = None,
        error: Optional[str] = None,
        content_length: int = 0,
        output_length: int = 0,
        correction_rounds: int = 0,
        verification_success: Optional[bool] = None,
        **metadata
    ) -> None:
        """Record the completion of a formalization operation.
        
        Args:
            success: Whether the operation was successful
            target_system: Target proof assistant system
            processing_time: Time taken in seconds
            error: Error message if failed
            content_length: Length of input content
            output_length: Length of generated output
            correction_rounds: Number of correction rounds
            verification_success: Whether verification succeeded
            **metadata: Additional metadata to record
        """
        with self._lock:
            # Update aggregate statistics
            self.stats["total_requests"] += 1
            
            if success:
                self.stats["successful_requests"] += 1
            else:
                self.stats["failed_requests"] += 1
                if error:
                    self.stats["error_counts"][error] += 1
            
            if processing_time:
                self.stats["total_processing_time"] += processing_time
            
            # Update system-specific statistics
            if target_system:
                system_stats = self.stats["by_system"][target_system]
                system_stats["requests"] += 1
                
                if success:
                    system_stats["successes"] += 1
                else:
                    system_stats["failures"] += 1
                
                if processing_time:
                    system_stats["total_processing_time"] += processing_time
                    system_stats["avg_processing_time"] = (
                        system_stats["total_processing_time"] / system_stats["requests"]
                    )
            
            # Update verification statistics
            if verification_success is not None:
                self.stats["verification_stats"]["attempted"] += 1
                if verification_success:
                    self.stats["verification_stats"]["successful"] += 1
                else:
                    self.stats["verification_stats"]["failed"] += 1
            
            # Record detailed event
            event = MetricEvent(
                name="formalization_completed",
                value=1,
                timestamp=datetime.now(),
                labels={
                    "success": str(success),
                    "target_system": target_system or "unknown",
                },
                metadata={
                    "processing_time": processing_time,
                    "error": error,
                    "content_length": content_length,
                    "output_length": output_length,
                    "correction_rounds": correction_rounds,
                    "verification_success": verification_success,
                    **metadata
                }
            )
            self.events.append(event)
            
            # Create processing metrics record
            proc_metrics = ProcessingMetrics(
                start_time=time.time() - (processing_time or 0),
                end_time=time.time(),
                success=success,
                error_message=error,
                target_system=target_system,
                content_length=content_length,
                output_length=output_length,
                correction_rounds=correction_rounds,
                verification_success=verification_success
            )
            self.processing_history.append(proc_metrics)
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self.active_requests.dec()
                
                status = "success" if success else "failure"
                system = target_system or "unknown"
                
                self.requests_total.labels(
                    target_system=system,
                    status=status
                ).inc()
                
                if processing_time:
                    self.processing_time_seconds.labels(
                        target_system=system
                    ).observe(processing_time)
                
                if output_length > 0:
                    self.output_length_bytes.labels(
                        target_system=system
                    ).observe(output_length)
                
                if correction_rounds > 0:
                    self.correction_rounds.labels(
                        target_system=system
                    ).observe(correction_rounds)
                
                if verification_success is not None:
                    result = "success" if verification_success else "failure"
                    self.verification_attempts.labels(
                        target_system=system,
                        result=result
                    ).inc()
                
                # Update success rate gauge
                if target_system:
                    system_stats = self.stats["by_system"][target_system]
                    if system_stats["requests"] > 0:
                        rate = system_stats["successes"] / system_stats["requests"]
                        self.success_rate.labels(target_system=target_system).set(rate)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics."""
        with self._lock:
            summary = dict(self.stats)
            
            # Add computed metrics
            total = summary["total_requests"]
            if total > 0:
                summary["overall_success_rate"] = summary["successful_requests"] / total
                summary["overall_failure_rate"] = summary["failed_requests"] / total
                summary["average_processing_time"] = summary["total_processing_time"] / total
            else:
                summary["overall_success_rate"] = 0.0
                summary["overall_failure_rate"] = 0.0
                summary["average_processing_time"] = 0.0
            
            # Add recent performance (last 100 operations)
            recent_ops = list(self.processing_history)[-100:]
            if recent_ops:
                recent_success_rate = sum(1 for op in recent_ops if op.success) / len(recent_ops)
                recent_avg_time = sum(op.processing_time for op in recent_ops) / len(recent_ops)
                
                summary["recent_success_rate"] = recent_success_rate
                summary["recent_average_processing_time"] = recent_avg_time
                summary["recent_operations_count"] = len(recent_ops)
            
            return summary
    
    def get_system_metrics(self, target_system: str) -> Dict[str, Any]:
        """Get metrics for a specific target system."""
        with self._lock:
            if target_system not in self.stats["by_system"]:
                return {"error": f"No data for system: {target_system}"}
            
            system_stats = dict(self.stats["by_system"][target_system])
            
            # Add computed metrics
            requests = system_stats["requests"]
            if requests > 0:
                system_stats["success_rate"] = system_stats["successes"] / requests
                system_stats["failure_rate"] = system_stats["failures"] / requests
            else:
                system_stats["success_rate"] = 0.0
                system_stats["failure_rate"] = 0.0
            
            return system_stats
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get detailed error analysis."""
        with self._lock:
            total_errors = sum(self.stats["error_counts"].values())
            
            error_analysis = {
                "total_errors": total_errors,
                "unique_error_types": len(self.stats["error_counts"]),
                "error_distribution": dict(self.stats["error_counts"]),
                "error_rates": {}
            }
            
            # Calculate error rates
            if total_errors > 0:
                for error, count in self.stats["error_counts"].items():
                    error_analysis["error_rates"][error] = count / total_errors
            
            return error_analysis
    
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metric events."""
        with self._lock:
            events = list(self.events)[-limit:]
            return [{
                "name": event.name,
                "value": event.value,
                "timestamp": event.timestamp.isoformat(),
                "labels": event.labels,
                "metadata": event.metadata
            } for event in events]
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        if not self.enable_prometheus:
            return "Prometheus metrics not enabled"
        
        from prometheus_client import generate_latest
        return generate_latest(self.registry).decode('utf-8')
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.events.clear()
            self.processing_history.clear()
            
            self.stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_processing_time": 0.0,
                "by_system": defaultdict(lambda: {
                    "requests": 0,
                    "successes": 0,
                    "failures": 0,
                    "avg_processing_time": 0.0,
                    "total_processing_time": 0.0
                }),
                "error_counts": defaultdict(int),
                "verification_stats": {
                    "attempted": 0,
                    "successful": 0,
                    "failed": 0
                }
            }
            
            if self.enable_prometheus:
                # Clear Prometheus metrics
                self._setup_prometheus_metrics()
    
    def export_csv(self, filepath: str) -> None:
        """Export processing history to CSV file."""
        import csv
        from pathlib import Path
        
        with self._lock:
            data = [metrics.to_dict() for metrics in self.processing_history]
        
        if not data:
            return
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        self.logger.info(f"Exported {len(data)} records to {filepath}")


# Global metrics instance
global_metrics = FormalizationMetrics()
