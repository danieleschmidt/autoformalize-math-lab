"""Health monitoring and system diagnostics.

This module provides comprehensive health monitoring capabilities
for the formalization pipeline and system components.
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

from .logging_config import setup_logger


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    check_func: Callable
    timeout: float = 10.0
    critical: bool = False
    enabled: bool = True
    interval: float = 60.0  # seconds
    last_run: Optional[float] = None
    last_status: Optional[HealthStatus] = None
    last_result: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    load_average: List[float]
    active_connections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_available_mb': self.memory_available_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'disk_free_gb': self.disk_free_gb,
            'load_average': self.load_average,
            'active_connections': self.active_connections
        }


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: float
    active_formalizations: int = 0
    completed_formalizations: int = 0
    failed_formalizations: int = 0
    average_processing_time: float = 0.0
    success_rate: float = 0.0
    cache_hit_rate: float = 0.0
    queue_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'active_formalizations': self.active_formalizations,
            'completed_formalizations': self.completed_formalizations,
            'failed_formalizations': self.failed_formalizations,
            'average_processing_time': self.average_processing_time,
            'success_rate': self.success_rate,
            'cache_hit_rate': self.cache_hit_rate,
            'queue_size': self.queue_size
        }


class SystemMonitor:
    """Monitors system resource usage and performance."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        
    def get_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024 * 1024 * 1024)
            
            # Load average
            load_average = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
            
            # Network connections
            connections = len(psutil.net_connections())
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                load_average=load_average,
                active_connections=connections
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                load_average=[0.0, 0.0, 0.0]
            )
    
    def check_system_health(self, metrics: SystemMetrics) -> HealthStatus:
        """Determine system health status based on metrics."""
        issues = []
        
        # Check CPU usage
        if metrics.cpu_percent > 90:
            issues.append("High CPU usage")
        elif metrics.cpu_percent > 70:
            issues.append("Elevated CPU usage")
        
        # Check memory usage
        if metrics.memory_percent > 90:
            issues.append("High memory usage")
        elif metrics.memory_percent > 80:
            issues.append("Elevated memory usage")
        
        # Check disk usage
        if metrics.disk_usage_percent > 95:
            issues.append("Critical disk usage")
        elif metrics.disk_usage_percent > 85:
            issues.append("High disk usage")
        
        # Check load average (if available)
        if metrics.load_average[0] > 10:
            issues.append("High system load")
        
        # Determine overall status
        if any("Critical" in issue or "High" in issue for issue in issues):
            if any("Critical" in issue for issue in issues):
                return HealthStatus.CRITICAL
            else:
                return HealthStatus.UNHEALTHY
        elif any("Elevated" in issue for issue in issues):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


class ApplicationMonitor:
    """Monitors application-specific metrics and health."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.metrics_history: List[ApplicationMetrics] = []
        self.max_history = 1000
        
        # Performance counters
        self.formalization_counter = 0
        self.success_counter = 0
        self.failure_counter = 0
        self.processing_times: List[float] = []
        
    def record_formalization(self, success: bool, processing_time: float):
        """Record a formalization attempt."""
        self.formalization_counter += 1
        
        if success:
            self.success_counter += 1
        else:
            self.failure_counter += 1
            
        self.processing_times.append(processing_time)
        
        # Keep only recent processing times
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
    
    def get_application_metrics(self) -> ApplicationMetrics:
        """Get current application metrics."""
        total_formalizations = self.formalization_counter
        success_rate = (self.success_counter / total_formalizations * 100) if total_formalizations > 0 else 0.0
        
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0
        
        metrics = ApplicationMetrics(
            timestamp=time.time(),
            active_formalizations=0,  # Would be tracked separately
            completed_formalizations=self.success_counter,
            failed_formalizations=self.failure_counter,
            average_processing_time=avg_processing_time,
            success_rate=success_rate,
            cache_hit_rate=0.0,  # Would be provided by cache system
            queue_size=0  # Would be provided by queue system
        )
        
        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
        
        return metrics
    
    def check_application_health(self, metrics: ApplicationMetrics) -> HealthStatus:
        """Determine application health status."""
        issues = []
        
        # Check success rate
        if metrics.success_rate < 50:
            issues.append("Low success rate")
        elif metrics.success_rate < 70:
            issues.append("Moderate success rate")
        
        # Check processing time
        if metrics.average_processing_time > 60:
            issues.append("Slow processing times")
        elif metrics.average_processing_time > 30:
            issues.append("Elevated processing times")
        
        # Check queue size
        if metrics.queue_size > 1000:
            issues.append("Large queue backlog")
        elif metrics.queue_size > 500:
            issues.append("Queue backlog")
        
        # Determine status
        if len(issues) >= 3 or any("Low" in issue or "Large" in issue for issue in issues):
            return HealthStatus.UNHEALTHY
        elif len(issues) >= 2:
            return HealthStatus.WARNING
        elif len(issues) >= 1:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.system_monitor = SystemMonitor()
        self.app_monitor = ApplicationMonitor()
        self.health_checks: Dict[str, HealthCheck] = {}
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Initialize built-in health checks
        self._initialize_health_checks()
        
        # Prometheus metrics (if available)
        if HAS_PROMETHEUS:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
    
    def _initialize_health_checks(self):
        """Initialize built-in health checks."""
        self.health_checks = {
            'system_resources': HealthCheck(
                name='system_resources',
                check_func=self._check_system_resources,
                critical=True,
                interval=30.0
            ),
            'application_performance': HealthCheck(
                name='application_performance',
                check_func=self._check_application_performance,
                critical=False,
                interval=60.0
            ),
            'disk_space': HealthCheck(
                name='disk_space',
                check_func=self._check_disk_space,
                critical=True,
                interval=120.0
            ),
            'memory_usage': HealthCheck(
                name='memory_usage',
                check_func=self._check_memory_usage,
                critical=True,
                interval=30.0
            )
        }
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        if not HAS_PROMETHEUS:
            return
            
        self.prom_metrics = {
            'system_cpu': Gauge('system_cpu_percent', 'CPU usage percentage', registry=self.registry),
            'system_memory': Gauge('system_memory_percent', 'Memory usage percentage', registry=self.registry),
            'system_disk': Gauge('system_disk_percent', 'Disk usage percentage', registry=self.registry),
            'app_success_rate': Gauge('app_success_rate', 'Application success rate', registry=self.registry),
            'app_processing_time': Histogram('app_processing_time', 'Processing time histogram', registry=self.registry),
            'formalization_total': Counter('formalization_total', 'Total formalizations', registry=self.registry),
            'formalization_success': Counter('formalization_success', 'Successful formalizations', registry=self.registry)
        }
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check overall system resource health."""
        metrics = self.system_monitor.get_system_metrics()
        status = self.system_monitor.check_system_health(metrics)
        
        return {
            'status': status.value,
            'metrics': metrics.to_dict(),
            'message': f"System health: {status.value}"
        }
    
    async def _check_application_performance(self) -> Dict[str, Any]:
        """Check application performance health."""
        metrics = self.app_monitor.get_application_metrics()
        status = self.app_monitor.check_application_health(metrics)
        
        return {
            'status': status.value,
            'metrics': metrics.to_dict(),
            'message': f"Application health: {status.value}"
        }
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability."""
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        
        if usage_percent > 95:
            status = HealthStatus.CRITICAL
            message = f"Critical: Disk usage at {usage_percent:.1f}%"
        elif usage_percent > 85:
            status = HealthStatus.UNHEALTHY
            message = f"Warning: Disk usage at {usage_percent:.1f}%"
        elif usage_percent > 75:
            status = HealthStatus.WARNING
            message = f"Caution: Disk usage at {usage_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage normal at {usage_percent:.1f}%"
        
        return {
            'status': status.value,
            'usage_percent': usage_percent,
            'free_gb': disk.free / (1024**3),
            'message': message
        }
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            status = HealthStatus.CRITICAL
            message = f"Critical: Memory usage at {memory.percent:.1f}%"
        elif memory.percent > 80:
            status = HealthStatus.UNHEALTHY
            message = f"High: Memory usage at {memory.percent:.1f}%"
        elif memory.percent > 70:
            status = HealthStatus.WARNING
            message = f"Elevated: Memory usage at {memory.percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage normal at {memory.percent:.1f}%"
        
        return {
            'status': status.value,
            'usage_percent': memory.percent,
            'available_mb': memory.available / (1024**2),
            'message': message
        }
    
    async def run_health_check(self, check_name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if check_name not in self.health_checks:
            raise ValueError(f"Unknown health check: {check_name}")
        
        check = self.health_checks[check_name]
        
        if not check.enabled:
            return {
                'name': check_name,
                'status': 'disabled',
                'message': 'Health check is disabled'
            }
        
        try:
            start_time = time.time()
            result = await asyncio.wait_for(check.check_func(), timeout=check.timeout)
            execution_time = time.time() - start_time
            
            check.last_run = time.time()
            check.last_status = HealthStatus(result['status'])
            check.last_result = result
            
            result.update({
                'name': check_name,
                'execution_time': execution_time,
                'timestamp': check.last_run
            })
            
            return result
            
        except asyncio.TimeoutError:
            error_result = {
                'name': check_name,
                'status': HealthStatus.UNHEALTHY.value,
                'message': f'Health check timed out after {check.timeout}s',
                'timestamp': time.time()
            }
            check.last_result = error_result
            return error_result
            
        except Exception as e:
            self.logger.error(f"Health check {check_name} failed: {e}")
            error_result = {
                'name': check_name,
                'status': HealthStatus.UNHEALTHY.value,
                'message': f'Health check error: {str(e)}',
                'timestamp': time.time()
            }
            check.last_result = error_result
            return error_result
    
    async def run_all_health_checks(self) -> Dict[str, Any]:
        """Run all enabled health checks."""
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for check_name, check in self.health_checks.items():
            if not check.enabled:
                continue
                
            result = await self.run_health_check(check_name)
            results[check_name] = result
            
            # Update overall status
            check_status = HealthStatus(result['status'])
            if check_status == HealthStatus.CRITICAL and check.critical:
                overall_status = HealthStatus.CRITICAL
            elif check_status == HealthStatus.UNHEALTHY and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.UNHEALTHY
            elif check_status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.WARNING
        
        return {
            'overall_status': overall_status.value,
            'timestamp': time.time(),
            'checks': results
        }
    
    async def start_monitoring(self, interval: float = 60.0):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        self.logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop continuous health monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self, base_interval: float):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Run health checks based on their individual intervals
                current_time = time.time()
                
                for check_name, check in self.health_checks.items():
                    if not check.enabled:
                        continue
                    
                    # Check if it's time to run this check
                    if (check.last_run is None or 
                        current_time - check.last_run >= check.interval):
                        
                        try:
                            await self.run_health_check(check_name)
                        except Exception as e:
                            self.logger.error(f"Error in monitoring check {check_name}: {e}")
                
                # Update Prometheus metrics if available
                if HAS_PROMETHEUS:
                    self._update_prometheus_metrics()
                
                await asyncio.sleep(base_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(base_interval)
    
    def _update_prometheus_metrics(self):
        """Update Prometheus metrics."""
        if not HAS_PROMETHEUS:
            return
        
        try:
            # System metrics
            system_metrics = self.system_monitor.get_system_metrics()
            self.prom_metrics['system_cpu'].set(system_metrics.cpu_percent)
            self.prom_metrics['system_memory'].set(system_metrics.memory_percent)
            self.prom_metrics['system_disk'].set(system_metrics.disk_usage_percent)
            
            # Application metrics
            app_metrics = self.app_monitor.get_application_metrics()
            self.prom_metrics['app_success_rate'].set(app_metrics.success_rate)
            
        except Exception as e:
            self.logger.error(f"Error updating Prometheus metrics: {e}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of current health status."""
        summary = {
            'timestamp': time.time(),
            'monitoring_active': self.monitoring_active,
            'checks_summary': {}
        }
        
        for check_name, check in self.health_checks.items():
            summary['checks_summary'][check_name] = {
                'enabled': check.enabled,
                'last_run': check.last_run,
                'last_status': check.last_status.value if check.last_status else None,
                'critical': check.critical
            }
        
        return summary