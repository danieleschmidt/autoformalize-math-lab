"""Enhanced resilience and error recovery system.

This module provides advanced error handling, retry mechanisms, circuit breakers,
and recovery strategies for the formalization pipeline.
"""

import asyncio
import time
import logging
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from enum import Enum
from functools import wraps

from ..utils.logging_config import setup_logger
from ..utils.metrics import FormalizationMetrics


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit tripped, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    
    
@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    expected_exception: Optional[type] = None


@dataclass
class HealthStatus:
    """System health status information."""
    is_healthy: bool = True
    timestamp: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    

class ResilienceManager:
    """Manages resilience patterns for the formalization system."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.metrics = FormalizationMetrics()
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.error_handlers: Dict[type, Callable] = {}
        
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> 'CircuitBreaker':
        """Register a circuit breaker."""
        circuit_breaker = CircuitBreaker(name, config, self.logger)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def register_health_check(self, name: str, check_fn: Callable) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_fn
    
    def register_error_handler(self, exception_type: type, handler_fn: Callable) -> None:
        """Register an error handler for specific exception types."""
        self.error_handlers[exception_type] = handler_fn
    
    async def check_health(self) -> HealthStatus:
        """Perform comprehensive health check."""
        status = HealthStatus()
        
        try:
            # Run all registered health checks
            for name, check_fn in self.health_checks.items():
                try:
                    result = await self._run_health_check(check_fn)
                    if not result:
                        status.is_healthy = False
                        status.errors.append(f"Health check '{name}' failed")
                except Exception as e:
                    status.is_healthy = False
                    status.errors.append(f"Health check '{name}' error: {str(e)}")
            
            # Check circuit breaker states
            for name, cb in self.circuit_breakers.items():
                if cb.state == CircuitState.OPEN:
                    status.warnings.append(f"Circuit breaker '{name}' is open")
            
            # Add system metrics
            status.metrics = {
                'circuit_breakers': len(self.circuit_breakers),
                'health_checks': len(self.health_checks),
                'open_circuits': sum(1 for cb in self.circuit_breakers.values() 
                                   if cb.state == CircuitState.OPEN)
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            status.is_healthy = False
            status.errors.append(f"Health check system error: {str(e)}")
        
        return status
    
    async def _run_health_check(self, check_fn: Callable) -> bool:
        """Run a single health check function."""
        if asyncio.iscoroutinefunction(check_fn):
            return await check_fn()
        else:
            return check_fn()
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        """Handle errors using registered handlers."""
        error_type = type(error)
        
        # Look for specific handler
        if error_type in self.error_handlers:
            try:
                return self.error_handlers[error_type](error, context)
            except Exception as handler_error:
                self.logger.error(f"Error handler failed: {handler_error}")
        
        # Look for base class handlers
        for registered_type, handler in self.error_handlers.items():
            if isinstance(error, registered_type):
                try:
                    return handler(error, context)
                except Exception as handler_error:
                    self.logger.error(f"Error handler failed: {handler_error}")
        
        # No handler found
        return None


class CircuitBreaker:
    """Circuit breaker implementation for external service calls."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig, logger: logging.Logger):
        self.name = name
        self.config = config
        self.logger = logger
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info(f"Circuit breaker '{self.name}' moving to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            self._record_success()
            return result
            
        except Exception as e:
            # Check if this is an expected exception type
            if (self.config.expected_exception is None or 
                isinstance(e, self.config.expected_exception)):
                self._record_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (time.time() - self.last_failure_time) >= self.config.timeout_seconds
    
    def _record_success(self) -> None:
        """Record successful call."""
        self.success_count += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.logger.info(f"Circuit breaker '{self.name}' recovering")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
    
    def _record_failure(self) -> None:
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker '{self.name}' tripped open")


class RetryManager:
    """Manages retry logic with exponential backoff and jitter."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.logger = setup_logger(__name__)
    
    def __call__(self, config: RetryConfig = None):
        """Decorator for retry functionality."""
        retry_config = config or self.config
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.retry(func, retry_config, *args, **kwargs)
            return wrapper
        return decorator
    
    async def retry(self, func: Callable, config: RetryConfig, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if attempt == config.max_attempts - 1:
                    # Final attempt failed
                    self.logger.error(f"All {config.max_attempts} retry attempts failed")
                    raise
                
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    config.base_delay * (config.exponential_base ** attempt),
                    config.max_delay
                )
                
                if config.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
                
                self.logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {str(e)}"
                )
                
                await asyncio.sleep(delay)
        
        # This should never be reached due to the raise in the loop
        raise last_exception


class GracefulDegradation:
    """Implements graceful degradation strategies."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.fallback_handlers: Dict[str, Callable] = {}
        
    def register_fallback(self, operation: str, handler: Callable) -> None:
        """Register a fallback handler for an operation."""
        self.fallback_handlers[operation] = handler
    
    async def execute_with_fallback(self, operation: str, primary_fn: Callable, 
                                  *args, **kwargs) -> Tuple[Any, bool]:
        """Execute primary function with fallback on failure."""
        try:
            # Try primary operation
            if asyncio.iscoroutinefunction(primary_fn):
                result = await primary_fn(*args, **kwargs)
            else:
                result = primary_fn(*args, **kwargs)
            return result, True
            
        except Exception as e:
            self.logger.warning(f"Primary operation '{operation}' failed: {e}")
            
            # Try fallback
            if operation in self.fallback_handlers:
                try:
                    fallback_fn = self.fallback_handlers[operation]
                    if asyncio.iscoroutinefunction(fallback_fn):
                        result = await fallback_fn(*args, **kwargs)
                    else:
                        result = fallback_fn(*args, **kwargs)
                    
                    self.logger.info(f"Fallback for '{operation}' succeeded")
                    return result, False
                    
                except Exception as fallback_error:
                    self.logger.error(f"Fallback for '{operation}' also failed: {fallback_error}")
                    raise
            else:
                self.logger.error(f"No fallback available for '{operation}'")
                raise


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class RetryExhaustedException(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


# Decorators for easy use
def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator for circuit breaker functionality."""
    config = config or CircuitBreakerConfig()
    logger = setup_logger("circuit_breaker")
    cb = CircuitBreaker(name, config, logger)
    return cb


def retry(config: RetryConfig = None):
    """Decorator for retry functionality."""
    retry_manager = RetryManager(config)
    return retry_manager(config)


# Global resilience manager instance
resilience_manager = ResilienceManager()