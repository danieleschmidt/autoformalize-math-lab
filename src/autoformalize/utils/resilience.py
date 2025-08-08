"""Resilience and reliability utilities.

This module provides utilities for making the formalization pipeline
robust against failures, including retry mechanisms, circuit breakers,
and graceful degradation strategies.
"""

import asyncio
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .logging_config import setup_logger


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
    retry_exceptions: tuple = (Exception,)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.logger = setup_logger(f"{__name__}.CircuitBreaker")
    
    def __call__(self, func):
        """Decorator to apply circuit breaker to a function."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.logger.info("Circuit breaker reset to CLOSED")
                
                return result
                
            except self.config.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
                
                raise e
        
        return wrapper


def retry_async(config: RetryConfig = None):
    """Async retry decorator with exponential backoff and jitter."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except config.retry_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter
                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        return wrapper
    
    return decorator


def retry_sync(config: RetryConfig = None):
    """Synchronous retry decorator."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retry_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter
                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    
    return decorator


class HealthCheck:
    """Health check utility for monitoring system health."""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], bool]] = {}
        self.logger = setup_logger(f"{__name__}.HealthCheck")
    
    def register(self, name: str, check_func: Callable[[], bool]):
        """Register a health check."""
        self.checks[name] = check_func
    
    async def check_all(self) -> Dict[str, bool]:
        """Run all health checks."""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                results[name] = bool(result)
            except Exception as e:
                self.logger.error(f"Health check '{name}' failed: {e}")
                results[name] = False
        
        return results
    
    async def is_healthy(self) -> bool:
        """Check if all systems are healthy."""
        results = await self.check_all()
        return all(results.values())


class GracefulDegradation:
    """Utility for implementing graceful degradation strategies."""
    
    def __init__(self):
        self.fallback_handlers: Dict[str, Callable] = {}
        self.logger = setup_logger(f"{__name__}.GracefulDegradation")
    
    def register_fallback(self, service: str, handler: Callable):
        """Register a fallback handler for a service."""
        self.fallback_handlers[service] = handler
    
    async def execute_with_fallback(
        self, 
        service: str, 
        primary_func: Callable, 
        *args, 
        **kwargs
    ):
        """Execute function with fallback on failure."""
        try:
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            else:
                return primary_func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Primary service '{service}' failed: {e}")
            
            if service in self.fallback_handlers:
                self.logger.info(f"Using fallback for service '{service}'")
                fallback = self.fallback_handlers[service]
                
                if asyncio.iscoroutinefunction(fallback):
                    return await fallback(*args, **kwargs)
                else:
                    return fallback(*args, **kwargs)
            else:
                self.logger.error(f"No fallback available for service '{service}'")
                raise e


# Global instances for convenience
health_check = HealthCheck()
graceful_degradation = GracefulDegradation()


def timeout_async(seconds: float):
    """Async timeout decorator."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
        return wrapper
    return decorator


def rate_limit(calls_per_second: float):
    """Rate limiting decorator."""
    min_interval = 1.0 / calls_per_second
    last_called = 0.0
    lock = asyncio.Lock()
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal last_called
            
            async with lock:
                now = time.time()
                elapsed = now - last_called
                
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)
                
                last_called = time.time()
            
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class ResourceMonitor:
    """Monitor system resources and enforce limits."""
    
    def __init__(self, max_memory_mb: int = 1024, max_cpu_percent: float = 80.0):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.logger = setup_logger(f"{__name__}.ResourceMonitor")
    
    def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage."""
        try:
            import psutil
            
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            return {
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'memory_limit_exceeded': memory_mb > self.max_memory_mb,
                'cpu_limit_exceeded': cpu_percent > self.max_cpu_percent,
            }
            
        except ImportError:
            self.logger.warning("psutil not available for resource monitoring")
            return {
                'memory_mb': 0,
                'cpu_percent': 0,
                'memory_limit_exceeded': False,
                'cpu_limit_exceeded': False,
            }
    
    def enforce_limits(self):
        """Enforce resource limits."""
        resources = self.check_resources()
        
        if resources['memory_limit_exceeded']:
            raise MemoryError(f"Memory limit exceeded: {resources['memory_mb']:.1f}MB > {self.max_memory_mb}MB")
        
        if resources['cpu_limit_exceeded']:
            self.logger.warning(f"CPU usage high: {resources['cpu_percent']:.1f}%")


# Global resource monitor
resource_monitor = ResourceMonitor()