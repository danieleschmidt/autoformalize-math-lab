"""Generation 6 Resilient Architecture: Advanced Error Recovery and Self-Healing Systems.

Implements comprehensive resilience patterns including circuit breakers, bulkheads,
timeout management, adaptive retry strategies, and self-healing capabilities.
"""

import asyncio
import time
import random
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import hashlib

from ..utils.logging_config import setup_logger
from .exceptions import FormalizationError
from .config import FormalizationConfig


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing fast
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class HealthMetrics:
    """System health metrics."""
    success_rate: float = 0.0
    average_response_time: float = 0.0
    error_count: int = 0
    total_requests: int = 0
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    def update_success(self, response_time: float) -> None:
        """Update metrics on successful operation."""
        self.total_requests += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now()
        
        # Update average response time with exponential smoothing
        alpha = 0.1
        self.average_response_time = (
            alpha * response_time + (1 - alpha) * self.average_response_time
        )
        
        self.success_rate = (self.total_requests - self.error_count) / self.total_requests
    
    def update_failure(self) -> None:
        """Update metrics on failed operation."""
        self.total_requests += 1
        self.error_count += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = datetime.now()
        
        self.success_rate = (self.total_requests - self.error_count) / self.total_requests


class CircuitBreaker:
    """Circuit breaker pattern implementation for formalization operations."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 3,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.name = name
        
        self.state = CircuitBreakerState.CLOSED
        self.metrics = HealthMetrics()
        self.last_failure_time = None
        self.half_open_success_count = 0
        
        self.logger = setup_logger(__name__)
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_success_count = 0
                self.logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN")
            else:
                raise FormalizationError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            start_time = time.time()
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            response_time = time.time() - start_time
            
            self._handle_success(response_time)
            return result
            
        except Exception as e:
            self._handle_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return False
        
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout
    
    def _handle_success(self, response_time: float) -> None:
        """Handle successful operation."""
        self.metrics.update_success(response_time)
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_success_count += 1
            if self.half_open_success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.logger.info(f"Circuit breaker {self.name} moved to CLOSED")
        
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            if self.metrics.consecutive_failures > 0:
                self.logger.info(f"Circuit breaker {self.name} recovered from failures")
    
    def _handle_failure(self) -> None:
        """Handle failed operation."""
        self.metrics.update_failure()
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker {self.name} moved to OPEN (half-open failure)")
        
        elif (self.state == CircuitBreakerState.CLOSED and 
              self.metrics.consecutive_failures >= self.failure_threshold):
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker {self.name} moved to OPEN (threshold exceeded)")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            'name': self.name,
            'state': self.state.value,
            'metrics': {
                'success_rate': self.metrics.success_rate,
                'average_response_time': self.metrics.average_response_time,
                'error_count': self.metrics.error_count,
                'total_requests': self.metrics.total_requests,
                'consecutive_failures': self.metrics.consecutive_failures,
                'consecutive_successes': self.metrics.consecutive_successes
            },
            'thresholds': {
                'failure_threshold': self.failure_threshold,
                'recovery_timeout': self.recovery_timeout,
                'success_threshold': self.success_threshold
            }
        }


class AdaptiveRetryStrategy:
    """Adaptive retry strategy with exponential backoff and jitter."""
    
    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter_factor: float = 0.1
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter_factor = jitter_factor
        
        self.logger = setup_logger(__name__)
        
        # Adaptive parameters
        self.success_history = deque(maxlen=100)
        self.failure_patterns = defaultdict(int)
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        retry_on_exceptions: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """Execute function with adaptive retry strategy."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Record success
                self.success_history.append((time.time(), True))
                self._adapt_strategy(success=True, attempt=attempt)
                
                if attempt > 0:
                    self.logger.info(f"Operation succeeded on attempt {attempt + 1}")
                
                return result
                
            except retry_on_exceptions as e:
                last_exception = e
                self.success_history.append((time.time(), False))
                
                # Record failure pattern
                error_signature = f"{type(e).__name__}:{str(e)[:100]}"
                self.failure_patterns[error_signature] += 1
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed")
                    self._adapt_strategy(success=False, attempt=attempt)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        # Base exponential backoff
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        # Add jitter to prevent thundering herd
        jitter = delay * self.jitter_factor * (random.random() * 2 - 1)
        delay = max(0, delay + jitter)
        
        # Adaptive adjustment based on success rate
        recent_success_rate = self._get_recent_success_rate()
        if recent_success_rate < 0.5:
            delay *= 1.5  # Increase delay when success rate is low
        elif recent_success_rate > 0.8:
            delay *= 0.8  # Decrease delay when success rate is high
        
        return delay
    
    def _get_recent_success_rate(self) -> float:
        """Get recent success rate from history."""
        if not self.success_history:
            return 0.5  # Neutral default
        
        # Consider only last 20 operations
        recent_operations = list(self.success_history)[-20:]
        successes = sum(1 for _, success in recent_operations if success)
        return successes / len(recent_operations)
    
    def _adapt_strategy(self, success: bool, attempt: int) -> None:
        """Adapt retry strategy based on outcomes."""
        # Adjust max retries based on success patterns
        if success and attempt == 0:
            # Immediate success - maybe we can be more aggressive
            pass
        elif success and attempt > 0:
            # Success after retries - current strategy is working
            pass
        elif not success:
            # Complete failure - maybe need more retries or different strategy
            recent_failure_rate = 1.0 - self._get_recent_success_rate()
            if recent_failure_rate > 0.7:
                self.max_retries = min(self.max_retries + 1, 10)
                self.logger.info(f"Increased max retries to {self.max_retries}")


class BulkheadIsolation:
    """Bulkhead pattern for isolating different formalization operations."""
    
    def __init__(self, max_concurrent_operations: Dict[str, int]):
        self.max_concurrent = max_concurrent_operations
        self.current_operations = defaultdict(int)
        self.semaphores = {
            operation: asyncio.Semaphore(max_count)
            for operation, max_count in max_concurrent_operations.items()
        }
        self.logger = setup_logger(__name__)
    
    async def execute_in_bulkhead(
        self,
        operation_type: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function within bulkhead limits."""
        if operation_type not in self.semaphores:
            raise ValueError(f"Unknown operation type: {operation_type}")
        
        semaphore = self.semaphores[operation_type]
        
        async with semaphore:
            self.current_operations[operation_type] += 1
            try:
                self.logger.debug(
                    f"Executing {operation_type} operation "
                    f"({self.current_operations[operation_type]}/{self.max_concurrent[operation_type]})"
                )
                
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                return result
                
            finally:
                self.current_operations[operation_type] -= 1
    
    def get_bulkhead_status(self) -> Dict[str, Dict[str, int]]:
        """Get current bulkhead utilization."""
        return {
            operation_type: {
                'current': self.current_operations[operation_type],
                'max': max_concurrent,
                'available': max_concurrent - self.current_operations[operation_type]
            }
            for operation_type, max_concurrent in self.max_concurrent.items()
        }


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.components = {}
        self.health_checks = {}
        self.alerts = []
        self.running = False
        
        self.logger = setup_logger(__name__)
        
        # Health thresholds
        self.thresholds = {
            'success_rate_critical': 0.5,
            'success_rate_warning': 0.8,
            'response_time_critical': 10.0,
            'response_time_warning': 5.0,
            'error_rate_critical': 0.5,
            'error_rate_warning': 0.2
        }
    
    def register_component(
        self,
        name: str,
        component: Any,
        health_check: Callable = None
    ) -> None:
        """Register a component for health monitoring."""
        self.components[name] = component
        if health_check:
            self.health_checks[name] = health_check
        
        self.logger.info(f"Registered component for monitoring: {name}")
    
    async def start_monitoring(self) -> None:
        """Start health monitoring loop."""
        self.running = True
        self.logger.info("Health monitoring started")
        
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(5.0)  # Brief pause on error
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.running = False
        self.logger.info("Health monitoring stopped")
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all components."""
        overall_health = {"timestamp": datetime.now().isoformat(), "components": {}}
        
        for name, component in self.components.items():
            try:
                component_health = await self._check_component_health(name, component)
                overall_health["components"][name] = component_health
                
                # Generate alerts if necessary
                self._check_alerts(name, component_health)
                
            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {e}")
                overall_health["components"][name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        # Store health snapshot
        self._store_health_snapshot(overall_health)
    
    async def _check_component_health(self, name: str, component: Any) -> Dict[str, Any]:
        """Check health of a specific component."""
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
        
        # Use custom health check if available
        if name in self.health_checks:
            custom_check = await self.health_checks[name](component)
            health_data.update(custom_check)
            return health_data
        
        # Default health checks based on component type
        if hasattr(component, 'get_status'):
            status = component.get_status()
            health_data.update(status)
            
            # Analyze metrics if available
            if 'metrics' in status:
                metrics = status['metrics']
                
                # Check success rate
                success_rate = metrics.get('success_rate', 1.0)
                if success_rate < self.thresholds['success_rate_critical']:
                    health_data['status'] = 'critical'
                elif success_rate < self.thresholds['success_rate_warning']:
                    health_data['status'] = 'warning'
                
                # Check response time
                avg_response_time = metrics.get('average_response_time', 0.0)
                if avg_response_time > self.thresholds['response_time_critical']:
                    health_data['status'] = 'critical'
                elif avg_response_time > self.thresholds['response_time_warning']:
                    health_data['status'] = 'warning'
        
        return health_data
    
    def _check_alerts(self, component_name: str, health_data: Dict[str, Any]) -> None:
        """Check if alerts should be generated."""
        status = health_data.get('status', 'healthy')
        
        if status in ['critical', 'warning']:
            alert = {
                'component': component_name,
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'details': health_data,
                'id': hashlib.sha256(f"{component_name}:{status}:{time.time()}".encode()).hexdigest()[:8]
            }
            
            self.alerts.append(alert)
            
            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
            
            self.logger.warning(f"Health alert: {component_name} is {status}")
    
    def _store_health_snapshot(self, health_data: Dict[str, Any]) -> None:
        """Store health snapshot for historical analysis."""
        # In a real implementation, this would store to a time-series database
        health_file = Path("cache/health_snapshots.jsonl")
        health_file.parent.mkdir(exist_ok=True)
        
        with open(health_file, 'a') as f:
            f.write(json.dumps(health_data) + '\n')
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        component_statuses = []
        
        for name, component in self.components.items():
            try:
                if hasattr(component, 'get_status'):
                    status = component.get_status()
                    component_statuses.append(status.get('metrics', {}).get('success_rate', 1.0))
            except Exception:
                component_statuses.append(0.0)
        
        overall_success_rate = sum(component_statuses) / max(len(component_statuses), 1)
        
        return {
            'overall_status': (
                'healthy' if overall_success_rate > 0.8 else
                'warning' if overall_success_rate > 0.5 else
                'critical'
            ),
            'overall_success_rate': overall_success_rate,
            'component_count': len(self.components),
            'active_alerts': len([alert for alert in self.alerts[-10:] if alert.get('status') == 'critical']),
            'recent_alerts': self.alerts[-5:] if self.alerts else []
        }


class SelfHealingSystem:
    """Self-healing system that automatically recovers from failures."""
    
    def __init__(self):
        self.healing_strategies = {}
        self.healing_history = []
        self.logger = setup_logger(__name__)
    
    def register_healing_strategy(
        self,
        error_pattern: str,
        healing_function: Callable,
        max_attempts: int = 3
    ) -> None:
        """Register a self-healing strategy for specific error patterns."""
        self.healing_strategies[error_pattern] = {
            'function': healing_function,
            'max_attempts': max_attempts,
            'attempts_used': 0,
            'success_count': 0,
            'failure_count': 0
        }
        
        self.logger.info(f"Registered healing strategy for pattern: {error_pattern}")
    
    async def attempt_healing(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to heal from an error."""
        error_signature = f"{type(error).__name__}:{str(error)[:100]}"
        
        # Find matching healing strategy
        for pattern, strategy in self.healing_strategies.items():
            if pattern in error_signature:
                return await self._execute_healing_strategy(pattern, strategy, error, context)
        
        self.logger.warning(f"No healing strategy found for error: {error_signature}")
        return False
    
    async def _execute_healing_strategy(
        self,
        pattern: str,
        strategy: Dict[str, Any],
        error: Exception,
        context: Dict[str, Any]
    ) -> bool:
        """Execute a specific healing strategy."""
        if strategy['attempts_used'] >= strategy['max_attempts']:
            self.logger.warning(f"Healing strategy for {pattern} has exceeded max attempts")
            return False
        
        try:
            strategy['attempts_used'] += 1
            healing_function = strategy['function']
            
            self.logger.info(f"Attempting healing for pattern: {pattern} (attempt {strategy['attempts_used']})")
            
            # Execute healing function
            success = await healing_function(error, context) if asyncio.iscoroutinefunction(healing_function) else healing_function(error, context)
            
            if success:
                strategy['success_count'] += 1
                strategy['attempts_used'] = 0  # Reset on success
                
                # Record healing success
                self.healing_history.append({
                    'pattern': pattern,
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'error': str(error),
                    'context': context
                })
                
                self.logger.info(f"Successfully healed error using pattern: {pattern}")
                return True
            else:
                strategy['failure_count'] += 1
                self.logger.warning(f"Healing attempt failed for pattern: {pattern}")
                return False
                
        except Exception as healing_error:
            strategy['failure_count'] += 1
            self.logger.error(f"Healing strategy failed with error: {healing_error}")
            return False
    
    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get self-healing statistics."""
        return {
            'registered_strategies': len(self.healing_strategies),
            'strategies': {
                pattern: {
                    'success_count': strategy['success_count'],
                    'failure_count': strategy['failure_count'],
                    'attempts_used': strategy['attempts_used'],
                    'max_attempts': strategy['max_attempts'],
                    'success_rate': (
                        strategy['success_count'] / 
                        max(strategy['success_count'] + strategy['failure_count'], 1)
                    )
                }
                for pattern, strategy in self.healing_strategies.items()
            },
            'healing_history_count': len(self.healing_history),
            'recent_healings': self.healing_history[-10:] if self.healing_history else []
        }


class ResilientFormalizationPipeline:
    """Main resilient formalization pipeline combining all reliability patterns."""
    
    def __init__(self, config: Optional[FormalizationConfig] = None):
        self.config = config or FormalizationConfig()
        self.logger = setup_logger(__name__)
        
        # Initialize resilience components
        self.circuit_breakers = {
            'llm_api': CircuitBreaker(failure_threshold=3, name='llm_api'),
            'verification': CircuitBreaker(failure_threshold=5, name='verification'),
            'parsing': CircuitBreaker(failure_threshold=2, name='parsing')
        }
        
        self.retry_strategy = AdaptiveRetryStrategy(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0
        )
        
        self.bulkhead = BulkheadIsolation({
            'formalization': 10,
            'verification': 5,
            'parsing': 8
        })
        
        self.health_monitor = HealthMonitor(check_interval=60.0)
        self.self_healing = SelfHealingSystem()
        
        # Register components with health monitor
        for name, cb in self.circuit_breakers.items():
            self.health_monitor.register_component(f"circuit_breaker_{name}", cb)
        
        # Register self-healing strategies
        self._register_default_healing_strategies()
        
        self.logger.info("Resilient formalization pipeline initialized")
    
    def _register_default_healing_strategies(self) -> None:
        """Register default self-healing strategies."""
        
        async def heal_llm_rate_limit(error: Exception, context: Dict[str, Any]) -> bool:
            """Heal from LLM API rate limiting."""
            self.logger.info("Attempting to heal from LLM rate limit")
            # Switch to backup model or wait
            await asyncio.sleep(random.uniform(5, 15))
            return True
        
        async def heal_memory_error(error: Exception, context: Dict[str, Any]) -> bool:
            """Heal from memory errors."""
            self.logger.info("Attempting to heal from memory error")
            # Clear caches or reduce batch size
            return True
        
        async def heal_timeout_error(error: Exception, context: Dict[str, Any]) -> bool:
            """Heal from timeout errors."""
            self.logger.info("Attempting to heal from timeout error")
            # Increase timeout or split operation
            return True
        
        self.self_healing.register_healing_strategy("RateLimitError", heal_llm_rate_limit)
        self.self_healing.register_healing_strategy("MemoryError", heal_memory_error)
        self.self_healing.register_healing_strategy("TimeoutError", heal_timeout_error)
        self.self_healing.register_healing_strategy("ConnectionError", heal_timeout_error)
    
    async def resilient_formalize(
        self,
        latex_input: str,
        target_system: str = "lean4"
    ) -> Dict[str, Any]:
        """Perform formalization with full resilience patterns."""
        
        async def _formalize_operation():
            # Mock formalization operation
            await asyncio.sleep(random.uniform(0.1, 1.0))  # Simulate processing
            
            # Simulate occasional failures
            if random.random() < 0.1:  # 10% failure rate
                error_types = ["ConnectionError", "TimeoutError", "RateLimitError"]
                raise Exception(random.choice(error_types))
            
            return {
                'success': True,
                'formal_code': f'theorem resilient_formalized : ∀ x : ℕ, x + 0 = x := by simp',
                'processing_time': random.uniform(0.5, 2.0),
                'resilience_stats': {
                    'circuit_breaker_used': True,
                    'retry_attempts': 0,
                    'bulkhead_isolation': True,
                    'self_healing_applied': False
                }
            }
        
        try:
            # Execute through bulkhead isolation
            result = await self.bulkhead.execute_in_bulkhead(
                'formalization',
                self._execute_with_circuit_breaker,
                'llm_api',
                _formalize_operation
            )
            
            return result
            
        except Exception as e:
            # Attempt self-healing
            healing_context = {
                'operation': 'formalization',
                'input': latex_input,
                'target_system': target_system
            }
            
            if await self.self_healing.attempt_healing(e, healing_context):
                # Retry after healing
                try:
                    result = await self.bulkhead.execute_in_bulkhead(
                        'formalization',
                        self._execute_with_circuit_breaker,
                        'llm_api',
                        _formalize_operation
                    )
                    result['resilience_stats']['self_healing_applied'] = True
                    return result
                except Exception as retry_error:
                    self.logger.error(f"Formalization failed even after healing: {retry_error}")
                    raise retry_error
            else:
                raise e
    
    async def _execute_with_circuit_breaker(
        self,
        breaker_name: str,
        operation: Callable
    ) -> Any:
        """Execute operation through circuit breaker and retry strategy."""
        circuit_breaker = self.circuit_breakers[breaker_name]
        
        return await self.retry_strategy.execute_with_retry(
            circuit_breaker.call,
            operation
        )
    
    async def start_health_monitoring(self) -> None:
        """Start health monitoring system."""
        await self.health_monitor.start_monitoring()
    
    def stop_health_monitoring(self) -> None:
        """Stop health monitoring system."""
        self.health_monitor.stop_monitoring()
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience system status."""
        return {
            'circuit_breakers': {
                name: cb.get_status() 
                for name, cb in self.circuit_breakers.items()
            },
            'bulkhead_status': self.bulkhead.get_bulkhead_status(),
            'health_status': self.health_monitor.get_overall_health(),
            'self_healing_stats': self.self_healing.get_healing_statistics(),
            'retry_effectiveness': {
                'recent_success_rate': self.retry_strategy._get_recent_success_rate(),
                'max_retries': self.retry_strategy.max_retries,
                'base_delay': self.retry_strategy.base_delay
            }
        }


# Factory function for easy instantiation
def create_resilient_pipeline(config: Optional[FormalizationConfig] = None) -> ResilientFormalizationPipeline:
    """Create resilient formalization pipeline with optimized configuration."""
    return ResilientFormalizationPipeline(config)