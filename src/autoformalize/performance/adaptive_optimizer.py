"""Adaptive performance optimization engine.

This module provides intelligent performance optimization including caching,
resource pooling, load balancing, and auto-scaling capabilities.
"""

import asyncio
import time
import json
import hashlib
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing
import psutil

from ..utils.logging_config import setup_logger
from ..utils.metrics import FormalizationMetrics


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    cache_hits: int = 0
    cache_misses: int = 0
    total_operations: int = 0
    average_latency: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    concurrent_operations: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class AdaptiveCache:
    """Intelligent caching system with automatic optimization."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()
        self.lock = threading.RLock()
        self.logger = setup_logger(__name__)
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def _generate_key(self, key: Any) -> str:
        """Generate cache key from any object."""
        if isinstance(key, str):
            return key
        return hashlib.sha256(str(key).encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache."""
        cache_key = self._generate_key(key)
        
        with self.lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check expiration
                if entry.is_expired():
                    del self.cache[cache_key]
                    self.access_order.remove(cache_key)
                    self.misses += 1
                    return None
                
                # Update access
                entry.access_count += 1
                entry.timestamp = time.time()
                
                # Move to end of access order
                self.access_order.remove(cache_key)
                self.access_order.append(cache_key)
                
                self.hits += 1
                return entry.value
            else:
                self.misses += 1
                return None
    
    def set(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Set item in cache."""
        cache_key = self._generate_key(key)
        ttl = ttl or self.default_ttl
        
        # Calculate size (approximation)
        try:
            size_bytes = len(pickle.dumps(value))
        except:
            size_bytes = 1024  # Default estimate
        
        with self.lock:
            # Remove if already exists
            if cache_key in self.cache:
                self.access_order.remove(cache_key)
            
            # Check if we need to evict
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes,
                ttl=ttl
            )
            
            self.cache[cache_key] = entry
            self.access_order.append(cache_key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self.access_order:
            lru_key = self.access_order.popleft()
            if lru_key in self.cache:
                del self.cache[lru_key]
                self.evictions += 1
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'total_size_bytes': sum(entry.size_bytes for entry in self.cache.values())
        }


class ResourcePool:
    """Adaptive resource pool for managing expensive operations."""
    
    def __init__(self, resource_factory: Callable, 
                 min_size: int = 2, max_size: int = 10):
        self.resource_factory = resource_factory
        self.min_size = min_size
        self.max_size = max_size
        
        self.available = asyncio.Queue()
        self.in_use = set()
        self.total_created = 0
        self.lock = asyncio.Lock()
        self.logger = setup_logger(__name__)
        
        # Initialize minimum resources
        asyncio.create_task(self._initialize())
    
    async def _initialize(self) -> None:
        """Initialize minimum resources."""
        for _ in range(self.min_size):
            resource = await self._create_resource()
            await self.available.put(resource)
    
    async def _create_resource(self) -> Any:
        """Create a new resource."""
        try:
            if asyncio.iscoroutinefunction(self.resource_factory):
                resource = await self.resource_factory()
            else:
                resource = self.resource_factory()
            
            self.total_created += 1
            return resource
        except Exception as e:
            self.logger.error(f"Failed to create resource: {e}")
            raise
    
    async def acquire(self, timeout: float = 30.0) -> Any:
        """Acquire a resource from the pool."""
        try:
            # Try to get available resource
            resource = await asyncio.wait_for(
                self.available.get(), timeout=1.0
            )
            
            async with self.lock:
                self.in_use.add(id(resource))
            
            return resource
            
        except asyncio.TimeoutError:
            # No available resources, try to create new one
            async with self.lock:
                total_resources = len(self.in_use) + self.available.qsize()
                
                if total_resources < self.max_size:
                    resource = await self._create_resource()
                    self.in_use.add(id(resource))
                    return resource
                else:
                    # Wait for available resource
                    resource = await asyncio.wait_for(
                        self.available.get(), timeout=timeout
                    )
                    self.in_use.add(id(resource))
                    return resource
    
    async def release(self, resource: Any) -> None:
        """Release a resource back to the pool."""
        async with self.lock:
            resource_id = id(resource)
            if resource_id in self.in_use:
                self.in_use.remove(resource_id)
                await self.available.put(resource)
            else:
                self.logger.warning("Releasing unknown resource")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'available': self.available.qsize(),
            'in_use': len(self.in_use),
            'total_created': self.total_created,
            'min_size': self.min_size,
            'max_size': self.max_size
        }


class LoadBalancer:
    """Simple load balancer for distributing work."""
    
    def __init__(self, workers: List[Any]):
        self.workers = workers
        self.current_index = 0
        self.lock = threading.Lock()
        self.worker_stats = defaultdict(lambda: {'requests': 0, 'errors': 0})
        self.logger = setup_logger(__name__)
    
    def get_next_worker(self) -> Any:
        """Get next worker using round-robin."""
        with self.lock:
            worker = self.workers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.workers)
            self.worker_stats[id(worker)]['requests'] += 1
            return worker
    
    def get_best_worker(self) -> Any:
        """Get worker with least load."""
        with self.lock:
            best_worker = min(
                self.workers,
                key=lambda w: self.worker_stats[id(w)]['requests']
            )
            self.worker_stats[id(best_worker)]['requests'] += 1
            return best_worker
    
    def report_error(self, worker: Any) -> None:
        """Report error for a worker."""
        self.worker_stats[id(worker)]['errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        total_requests = sum(stats['requests'] for stats in self.worker_stats.values())
        total_errors = sum(stats['errors'] for stats in self.worker_stats.values())
        
        return {
            'workers': len(self.workers),
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate': total_errors / total_requests if total_requests > 0 else 0.0,
            'worker_stats': dict(self.worker_stats)
        }


class AutoScaler:
    """Automatic scaling based on system metrics."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 20):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        
        self.cpu_threshold_up = 80.0
        self.cpu_threshold_down = 30.0
        self.memory_threshold_up = 80.0
        
        self.metrics_history = deque(maxlen=10)
        self.last_scale_time = 0
        self.scale_cooldown = 60  # seconds
        
        self.logger = setup_logger(__name__)
    
    def should_scale_up(self) -> bool:
        """Check if we should scale up."""
        if len(self.metrics_history) < 3:
            return False
        
        # Check recent CPU usage
        recent_cpu = [m['cpu_percent'] for m in list(self.metrics_history)[-3:]]
        avg_cpu = sum(recent_cpu) / len(recent_cpu)
        
        # Check memory usage
        memory_percent = psutil.virtual_memory().percent
        
        scale_up = (
            avg_cpu > self.cpu_threshold_up or
            memory_percent > self.memory_threshold_up
        ) and self.current_workers < self.max_workers
        
        return scale_up and self._can_scale()
    
    def should_scale_down(self) -> bool:
        """Check if we should scale down."""
        if len(self.metrics_history) < 5:
            return False
        
        # Check recent CPU usage
        recent_cpu = [m['cpu_percent'] for m in list(self.metrics_history)[-5:]]
        avg_cpu = sum(recent_cpu) / len(recent_cpu)
        
        scale_down = (
            avg_cpu < self.cpu_threshold_down and
            self.current_workers > self.min_workers
        )
        
        return scale_down and self._can_scale()
    
    def _can_scale(self) -> bool:
        """Check if enough time has passed since last scaling."""
        return (time.time() - self.last_scale_time) > self.scale_cooldown
    
    def record_metrics(self) -> None:
        """Record current system metrics."""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'active_connections': len(psutil.net_connections())
        }
        
        self.metrics_history.append(metrics)
    
    def scale_up(self) -> int:
        """Scale up workers."""
        if self.should_scale_up():
            new_count = min(self.current_workers + 1, self.max_workers)
            self.current_workers = new_count
            self.last_scale_time = time.time()
            self.logger.info(f"Scaled up to {new_count} workers")
            return new_count
        return self.current_workers
    
    def scale_down(self) -> int:
        """Scale down workers."""
        if self.should_scale_down():
            new_count = max(self.current_workers - 1, self.min_workers)
            self.current_workers = new_count
            self.last_scale_time = time.time()
            self.logger.info(f"Scaled down to {new_count} workers")
            return new_count
        return self.current_workers
    
    def get_recommended_workers(self) -> int:
        """Get recommended number of workers."""
        self.record_metrics()
        
        if self.should_scale_up():
            return self.scale_up()
        elif self.should_scale_down():
            return self.scale_down()
        
        return self.current_workers


class ConcurrentExecutor:
    """High-performance concurrent execution manager."""
    
    def __init__(self, max_threads: int = 10, max_processes: int = 4):
        self.max_threads = max_threads
        self.max_processes = max_processes
        
        self.thread_executor = ThreadPoolExecutor(max_workers=max_threads)
        self.process_executor = ProcessPoolExecutor(max_workers=max_processes)
        
        self.active_tasks = set()
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        self.logger = setup_logger(__name__)
    
    async def execute_async(self, tasks: List[Callable], 
                          use_processes: bool = False) -> List[Any]:
        """Execute tasks concurrently."""
        executor = self.process_executor if use_processes else self.thread_executor
        loop = asyncio.get_event_loop()
        
        # Submit all tasks
        futures = []
        for task in tasks:
            future = loop.run_in_executor(executor, task)
            futures.append(future)
            self.active_tasks.add(future)
        
        # Wait for completion
        results = []
        for future in asyncio.as_completed(futures):
            try:
                result = await future
                results.append(result)
                self.completed_tasks += 1
            except Exception as e:
                self.logger.error(f"Task failed: {e}")
                results.append(None)
                self.failed_tasks += 1
            finally:
                self.active_tasks.discard(future)
        
        return results
    
    async def batch_execute(self, items: List[Any], process_fn: Callable,
                          batch_size: int = 10) -> List[Any]:
        """Execute items in batches for better resource management."""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [lambda item=item: process_fn(item) for item in batch]
            
            batch_results = await self.execute_async(batch_tasks)
            results.extend(batch_results)
            
            # Small delay between batches to prevent resource exhaustion
            await asyncio.sleep(0.1)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total_tasks = self.completed_tasks + self.failed_tasks
        success_rate = self.completed_tasks / total_tasks if total_tasks > 0 else 1.0
        
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': success_rate,
            'thread_workers': self.max_threads,
            'process_workers': self.max_processes
        }
    
    def shutdown(self) -> None:
        """Shutdown executors gracefully."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.cache = AdaptiveCache()
        self.auto_scaler = AutoScaler()
        self.executor = ConcurrentExecutor()
        
        self.metrics = PerformanceMetrics()
        self.optimization_history = deque(maxlen=100)
        
        self.logger = setup_logger(__name__)
        
        # Start background optimization
        self.optimization_task = None
        self._optimization_enabled = True
    
    def start_optimization_loop(self) -> None:
        """Start background optimization loop."""
        try:
            if self.optimization_task is None or self.optimization_task.done():
                self.optimization_task = asyncio.create_task(self._optimization_loop())
        except RuntimeError:
            # No event loop running, will start later
            self.logger.debug("No event loop available, optimization loop will start later")
    
    async def _optimization_loop(self) -> None:
        """Background optimization loop."""
        while True:
            try:
                await self.optimize()
                await asyncio.sleep(60)  # Optimize every minute
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(300)  # Wait longer on error
    
    async def optimize(self) -> Dict[str, Any]:
        """Perform optimization based on current metrics."""
        optimization_start = time.time()
        
        # Collect current metrics
        self._update_metrics()
        
        optimizations = {
            'timestamp': optimization_start,
            'cache_optimized': False,
            'scaling_adjusted': False,
            'memory_cleaned': False
        }
        
        # Cache optimization
        cache_stats = self.cache.get_stats()
        if cache_stats['hit_rate'] < 0.5 and cache_stats['size'] > 0:
            # Poor hit rate, clear old entries
            self.cache.clear()
            optimizations['cache_optimized'] = True
            self.logger.info("Cache cleared due to poor hit rate")
        
        # Auto-scaling
        recommended_workers = self.auto_scaler.get_recommended_workers()
        current_workers = self.auto_scaler.current_workers
        if recommended_workers != current_workers:
            optimizations['scaling_adjusted'] = True
            self.logger.info(f"Scaling adjusted: {current_workers} -> {recommended_workers}")
        
        # Memory cleanup
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 85:
            import gc
            gc.collect()
            optimizations['memory_cleaned'] = True
            self.logger.info("Memory cleanup performed")
        
        optimization_time = time.time() - optimization_start
        optimizations['optimization_time'] = optimization_time
        
        self.optimization_history.append(optimizations)
        return optimizations
    
    def _update_metrics(self) -> None:
        """Update performance metrics."""
        cache_stats = self.cache.get_stats()
        executor_stats = self.executor.get_stats()
        
        self.metrics.cache_hits = cache_stats['hits']
        self.metrics.cache_misses = cache_stats['misses']
        self.metrics.memory_usage_mb = psutil.virtual_memory().used / (1024 * 1024)
        self.metrics.cpu_usage_percent = psutil.cpu_percent()
        self.metrics.concurrent_operations = executor_stats['active_tasks']
        
        total_operations = executor_stats['completed_tasks'] + executor_stats['failed_tasks']
        self.metrics.total_operations = total_operations
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'metrics': {
                'cache_hit_rate': self.metrics.cache_hit_rate,
                'total_operations': self.metrics.total_operations,
                'memory_usage_mb': self.metrics.memory_usage_mb,
                'cpu_usage_percent': self.metrics.cpu_usage_percent,
                'concurrent_operations': self.metrics.concurrent_operations
            },
            'cache': self.cache.get_stats(),
            'auto_scaler': {
                'current_workers': self.auto_scaler.current_workers,
                'min_workers': self.auto_scaler.min_workers,
                'max_workers': self.auto_scaler.max_workers,
                'metrics_history_size': len(self.auto_scaler.metrics_history)
            },
            'executor': self.executor.get_stats(),
            'optimization_history_size': len(self.optimization_history),
            'last_optimization': self.optimization_history[-1] if self.optimization_history else None
        }
    
    def shutdown(self) -> None:
        """Shutdown optimizer gracefully."""
        if self.optimization_task:
            self.optimization_task.cancel()
        self.executor.shutdown()


# Global performance optimizer instance (initialized on first use)
performance_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get or create the global performance optimizer instance."""
    global performance_optimizer
    if performance_optimizer is None:
        performance_optimizer = PerformanceOptimizer()
    return performance_optimizer