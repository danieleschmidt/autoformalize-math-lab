"""Advanced error handling and resilience utilities.

This module provides comprehensive error handling, circuit breakers,
retry mechanisms, and failure recovery for the formalization pipeline.
"""

import asyncio
import functools
import logging
import time
import random
from typing import (
    Any, Callable, Dict, List, Optional, Type, Union, 
    Awaitable, TypeVar, Generic
)
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

from .logging_config import setup_logger


T = TypeVar('T')


class FailureType(Enum):
    """Types of failures that can occur."""
    NETWORK_ERROR = "network_error"
    API_RATE_LIMIT = "api_rate_limit"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    PARSING_ERROR = "parsing_error"
    VERIFICATION_ERROR = "verification_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # linear, exponential, fibonacci
    retry_on: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    stop_on: List[Type[Exception]] = field(default_factory=list)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    name: Optional[str] = None


@dataclass
class FailureRecord:
    """Record of a failure occurrence."""
    timestamp: float
    failure_type: FailureType
    exception_type: str
    error_message: str
    context: Dict[str, Any] = field(default_factory=dict)
    retry_attempt: int = 0
    duration: float = 0.0


class AdvancedErrorHandler:
    """Advanced error handling with circuit breakers and retry logic."""
    
    def __init__(self):
        """Initialize error handler."""
        self.logger = setup_logger(__name__)
        self.failure_history: List[FailureRecord] = []
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self.failure_stats: Dict[str, int] = defaultdict(int)
        
    def classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure based on exception."""
        exception_name = type(exception).__name__.lower()
        error_message = str(exception).lower()
        
        # Network-related errors
        if any(term in exception_name for term in ['connection', 'network', 'socket', 'timeout']):
            if 'timeout' in exception_name or 'timeout' in error_message:
                return FailureType.TIMEOUT_ERROR
            return FailureType.NETWORK_ERROR
        
        # API rate limiting
        if any(term in error_message for term in ['rate limit', 'quota', 'too many requests']):
            return FailureType.API_RATE_LIMIT
        
        # Validation errors
        if any(term in exception_name for term in ['validation', 'schema', 'format']):
            return FailureType.VALIDATION_ERROR
        
        # Parsing errors
        if any(term in exception_name for term in ['parse', 'syntax', 'decode']):
            return FailureType.PARSING_ERROR
        
        # Verification errors
        if any(term in error_message for term in ['verification', 'proof', 'lean', 'isabelle', 'coq']):
            return FailureType.VERIFICATION_ERROR
        
        # Resource exhaustion
        if any(term in error_message for term in ['memory', 'disk', 'cpu', 'resource']):
            return FailureType.RESOURCE_EXHAUSTION
        
        # System errors
        if any(term in exception_name for term in ['system', 'os', 'file', 'permission']):
            return FailureType.SYSTEM_ERROR
        
        return FailureType.UNKNOWN_ERROR
    
    def record_failure(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        retry_attempt: int = 0,
        duration: float = 0.0
    ) -> FailureRecord:
        """Record a failure for analysis."""
        failure_type = self.classify_failure(exception)
        
        record = FailureRecord(
            timestamp=time.time(),
            failure_type=failure_type,
            exception_type=type(exception).__name__,
            error_message=str(exception),
            context=context or {},
            retry_attempt=retry_attempt,
            duration=duration
        )
        
        self.failure_history.append(record)
        self.failure_stats[failure_type.value] += 1
        
        # Limit history size
        if len(self.failure_history) > 1000:
            self.failure_history = self.failure_history[-500:]
        
        self.logger.warning(
            f"Failure recorded: {failure_type.value} - {record.error_message[:100]}"
        )
        
        return record
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> 'CircuitBreaker':
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                config=config or CircuitBreakerConfig(name=name),
                error_handler=self
            )
        return self.circuit_breakers[name]
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get summary of failure statistics."""
        if not self.failure_history:
            return {
                'total_failures': 0,
                'by_type': {},
                'recent_failures': [],
                'failure_rate': 0.0
            }
        
        # Recent failures (last hour)
        current_time = time.time()
        recent_failures = [
            f for f in self.failure_history
            if current_time - f.timestamp < 3600
        ]
        
        # Failure rate calculation
        if self.failure_history:
            time_span = current_time - self.failure_history[0].timestamp
            failure_rate = len(self.failure_history) / max(time_span / 3600, 1)  # Per hour
        else:
            failure_rate = 0.0
        
        return {
            'total_failures': len(self.failure_history),
            'by_type': dict(self.failure_stats),
            'recent_failures': len(recent_failures),
            'failure_rate': failure_rate,
            'circuit_breakers': {
                name: cb.get_status() for name, cb in self.circuit_breakers.items()
            }
        }


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig, error_handler: Optional[AdvancedErrorHandler] = None):
        """Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
            error_handler: Optional error handler for logging
        """
        self.config = config
        self.error_handler = error_handler
        self.logger = setup_logger(__name__)
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = time.time()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.logger.info(f"Circuit breaker {self.config.name} entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.config.name} is OPEN")
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - reset failure count
            self.on_success()
            return result
            
        except self.config.expected_exception as e:
            self.on_failure(e)
            raise
    
    def on_success(self) -> None:
        """Handle successful execution."""
        self.failure_count = 0
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.logger.info(f"Circuit breaker {self.config.name} closed after successful execution")
    
    def on_failure(self, exception: Exception) -> None:
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.error_handler:
            self.error_handler.record_failure(
                exception,
                context={'circuit_breaker': self.config.name}
            )
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(
                f"Circuit breaker {self.config.name} opened after {self.failure_count} failures"
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time,
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout
            }
        }


class RetryHandler:
    """Advanced retry handler with multiple backoff strategies."""
    
    def __init__(self, config: Optional[RetryConfig] = None, error_handler: Optional[AdvancedErrorHandler] = None):
        """Initialize retry handler.
        
        Args:
            config: Retry configuration
            error_handler: Optional error handler for logging
        """
        self.config = config or RetryConfig()
        self.error_handler = error_handler
        self.logger = setup_logger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to function."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.retry_call(func, *args, **kwargs)
        return wrapper
    
    async def retry_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                start_time = time.time()
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success
                if attempt > 0:
                    self.logger.info(f"Function succeeded after {attempt + 1} attempts")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                last_exception = e
                
                # Check if we should stop retrying
                if any(isinstance(e, exc_type) for exc_type in self.config.stop_on):
                    self.logger.info(f"Stopping retry due to {type(e).__name__}")
                    break
                
                # Check if we should retry this exception
                if not any(isinstance(e, exc_type) for exc_type in self.config.retry_on):
                    self.logger.info(f"Not retrying {type(e).__name__}")
                    break
                
                # Record failure
                if self.error_handler:
                    self.error_handler.record_failure(
                        e,
                        context={'retry_attempt': attempt + 1},
                        retry_attempt=attempt + 1,
                        duration=execution_time
                    )
                
                # Check if we have more attempts
                if attempt == self.config.max_attempts - 1:
                    self.logger.error(f"All {self.config.max_attempts} retry attempts failed")
                    break
                
                # Calculate delay
                delay = self.calculate_delay(attempt)
                
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)[:100]}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        raise last_exception
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.backoff_strategy == "linear":
            delay = self.config.base_delay * (attempt + 1)
        elif self.config.backoff_strategy == "exponential":
            delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        elif self.config.backoff_strategy == "fibonacci":
            delay = self.config.base_delay * self._fibonacci(attempt + 1)
        else:
            delay = self.config.base_delay
        
        # Apply jitter
        if self.config.jitter:
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor
        
        # Respect max delay
        delay = min(delay, self.config.max_delay)
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


class GracefulDegradation:
    """Graceful degradation for handling partial failures."""
    
    def __init__(self, error_handler: Optional[AdvancedErrorHandler] = None):
        """Initialize graceful degradation handler.
        
        Args:
            error_handler: Optional error handler for logging
        """
        self.error_handler = error_handler
        self.logger = setup_logger(__name__)
        self.fallback_strategies: Dict[str, Callable] = {}
    
    def register_fallback(self, operation: str, fallback_func: Callable) -> None:
        """Register fallback strategy for operation.
        
        Args:
            operation: Operation name
            fallback_func: Fallback function to call
        """
        self.fallback_strategies[operation] = fallback_func
        self.logger.info(f"Registered fallback for operation: {operation}")
    
    async def execute_with_fallback(
        self,
        operation: str,
        primary_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with graceful degradation.
        
        Args:
            operation: Operation name
            primary_func: Primary function to execute
            *args: Arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Result from primary function or fallback
        """
        try:
            # Try primary function
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            else:
                return primary_func(*args, **kwargs)
                
        except Exception as e:
            self.logger.warning(f"Primary function failed for {operation}: {str(e)[:100]}")
            
            if self.error_handler:
                self.error_handler.record_failure(
                    e,
                    context={'operation': operation, 'degraded': True}
                )
            
            # Try fallback
            if operation in self.fallback_strategies:
                self.logger.info(f"Using fallback strategy for {operation}")
                try:
                    fallback_func = self.fallback_strategies[operation]
                    if asyncio.iscoroutinefunction(fallback_func):
                        return await fallback_func(*args, **kwargs)
                    else:
                        return fallback_func(*args, **kwargs)
                        
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed for {operation}: {str(fallback_error)[:100]}")
                    if self.error_handler:
                        self.error_handler.record_failure(
                            fallback_error,
                            context={'operation': operation, 'fallback_failed': True}
                        )
                    raise
            else:
                self.logger.error(f"No fallback strategy registered for {operation}")
                raise


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic to functions."""
    def decorator(func: Callable) -> Callable:
        retry_handler = RetryHandler(config)
        return retry_handler(func)
    return decorator


def with_circuit_breaker(config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker to functions."""
    def decorator(func: Callable) -> Callable:
        circuit_breaker = CircuitBreaker(config or CircuitBreakerConfig())
        return circuit_breaker(func)
    return decorator


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class MaxRetriesExceededError(Exception):
    """Raised when maximum retry attempts are exceeded."""
    pass


# Global error handler instance
global_error_handler = AdvancedErrorHandler()


def get_global_error_handler() -> AdvancedErrorHandler:
    """Get the global error handler instance."""
    return global_error_handler