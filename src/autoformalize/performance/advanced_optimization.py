"""Advanced performance optimization and scaling system.

This module provides sophisticated optimization techniques including
adaptive caching, parallel processing, load balancing, and intelligent
resource management for large-scale mathematical formalization.
"""

import asyncio
import time
import hashlib
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict
import threading
from pathlib import Path
import multiprocessing as mp

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    redis = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from ..utils.logging_config import setup_logger
from ..utils.caching import CacheManager


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"


class ScalingMode(Enum):
    """Scaling modes for processing."""
    SINGLE_THREAD = "single"
    MULTI_THREAD = "threading"
    MULTI_PROCESS = "multiprocessing"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    operation_count: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_ratio: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationConfig:
    """Configuration for optimization system."""
    strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE
    scaling_mode: ScalingMode = ScalingMode.HYBRID
    max_workers: int = field(default_factory=lambda: mp.cpu_count())
    cache_size_mb: int = 512
    batch_size: int = 10
    enable_compression: bool = True
    enable_persistent_cache: bool = True
    optimization_interval: float = 60.0
    performance_target_rps: float = 100.0
    memory_limit_mb: int = 2048


class AdaptiveCache:
    """Adaptive caching system with intelligent eviction."""
    
    def __init__(
        self,
        max_size_mb: int = 512,
        enable_compression: bool = True,
        enable_persistence: bool = True
    ):
        """Initialize adaptive cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
            enable_compression: Whether to compress cached data
            enable_persistence: Whether to persist cache to disk
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.enable_compression = enable_compression
        self.enable_persistence = enable_persistence
        
        self.logger = setup_logger(__name__)
        
        # Multi-level cache
        self.l1_cache: Dict[str, Any] = {}  # Hot cache
        self.l2_cache: Dict[str, Any] = {}  # Warm cache
        self.l3_cache: Dict[str, Any] = {}  # Cold cache
        
        # Cache metadata
        self.access_frequency: Dict[str, int] = defaultdict(int)
        self.access_recency: Dict[str, float] = {}
        self.cache_sizes: Dict[str, int] = {}
        self.total_size = 0
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Background optimization
        self._optimization_task = None
        self._start_background_optimization()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache with adaptive promotion."""
        # Check L1 cache first
        if key in self.l1_cache:
            self.hits += 1
            self.access_frequency[key] += 1
            self.access_recency[key] = time.time()
            return self.l1_cache[key]
        
        # Check L2 cache
        if key in self.l2_cache:
            self.hits += 1
            self.access_frequency[key] += 1
            self.access_recency[key] = time.time()
            
            # Promote to L1 if frequently accessed
            if self.access_frequency[key] > 5:
                await self._promote_to_l1(key, self.l2_cache[key])
                del self.l2_cache[key]
            
            return self.l2_cache[key]
        
        # Check L3 cache
        if key in self.l3_cache:
            self.hits += 1
            self.access_frequency[key] += 1
            self.access_recency[key] = time.time()
            
            # Promote to L2
            await self._promote_to_l2(key, self.l3_cache[key])
            del self.l3_cache[key]
            
            return self.l3_cache[key]
        
        self.misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set item in cache with intelligent placement."""
        # Calculate item size
        item_size = self._calculate_size(value)
        
        # Ensure cache capacity
        await self._ensure_capacity(item_size)
        
        # Place in L1 cache initially
        self.l1_cache[key] = value
        self.cache_sizes[key] = item_size
        self.total_size += item_size
        self.access_frequency[key] = 1
        self.access_recency[key] = time.time()
        
        # Set TTL if specified
        if ttl:
            asyncio.create_task(self._expire_key(key, ttl))
    
    async def _promote_to_l1(self, key: str, value: Any) -> None:
        """Promote item to L1 cache."""
        # Ensure L1 capacity
        l1_target_size = self.max_size_bytes // 4  # L1 is 25% of total
        current_l1_size = sum(self.cache_sizes.get(k, 0) for k in self.l1_cache)
        
        item_size = self.cache_sizes.get(key, self._calculate_size(value))
        
        if current_l1_size + item_size > l1_target_size:
            await self._evict_from_l1()
        
        self.l1_cache[key] = value
    
    async def _promote_to_l2(self, key: str, value: Any) -> None:
        """Promote item to L2 cache."""
        l2_target_size = self.max_size_bytes // 2  # L2 is 50% of total
        current_l2_size = sum(self.cache_sizes.get(k, 0) for k in self.l2_cache)
        
        item_size = self.cache_sizes.get(key, self._calculate_size(value))
        
        if current_l2_size + item_size > l2_target_size:
            await self._evict_from_l2()
        
        self.l2_cache[key] = value
    
    async def _ensure_capacity(self, required_size: int) -> None:
        """Ensure sufficient cache capacity."""
        while self.total_size + required_size > self.max_size_bytes:
            # Evict from least important cache level
            if self.l3_cache:
                await self._evict_from_l3()
            elif self.l2_cache:
                await self._evict_from_l2()
            elif self.l1_cache:
                await self._evict_from_l1()
            else:
                break
    
    async def _evict_from_l1(self) -> None:
        """Evict least valuable item from L1 cache."""
        if not self.l1_cache:
            return
        
        # Use LFU + LRU hybrid eviction
        lru_key = min(self.l1_cache.keys(), 
                     key=lambda k: (self.access_frequency[k], self.access_recency[k]))
        
        value = self.l1_cache[lru_key]
        del self.l1_cache[lru_key]
        
        # Demote to L2
        await self._promote_to_l2(lru_key, value)
        self.evictions += 1
    
    async def _evict_from_l2(self) -> None:
        """Evict least valuable item from L2 cache."""
        if not self.l2_cache:
            return
        
        lru_key = min(self.l2_cache.keys(),
                     key=lambda k: (self.access_frequency[k], self.access_recency[k]))
        
        value = self.l2_cache[lru_key]
        del self.l2_cache[lru_key]
        
        # Demote to L3
        self.l3_cache[lru_key] = value
        self.evictions += 1
    
    async def _evict_from_l3(self) -> None:
        """Evict item from L3 cache (complete removal)."""
        if not self.l3_cache:
            return
        
        lru_key = min(self.l3_cache.keys(),
                     key=lambda k: (self.access_frequency[k], self.access_recency[k]))
        
        # Complete removal
        item_size = self.cache_sizes.get(lru_key, 0)
        self.total_size -= item_size
        
        del self.l3_cache[lru_key]
        del self.cache_sizes[lru_key]
        del self.access_frequency[lru_key]
        del self.access_recency[lru_key]
        
        self.evictions += 1
    
    async def _expire_key(self, key: str, ttl: float) -> None:
        """Expire key after TTL."""
        await asyncio.sleep(ttl)
        
        # Remove from all cache levels
        for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
            if key in cache:
                del cache[key]
                break
        
        # Clean up metadata
        if key in self.cache_sizes:
            self.total_size -= self.cache_sizes[key]
            del self.cache_sizes[key]
            del self.access_frequency[key]
            del self.access_recency[key]
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of cached value."""
        if isinstance(value, str):
            return len(value.encode('utf-8'))
        elif isinstance(value, (int, float)):
            return 8
        elif isinstance(value, dict):
            return len(json.dumps(value).encode('utf-8'))
        elif isinstance(value, list):
            return sum(self._calculate_size(item) for item in value)
        else:
            # Fallback: use string representation
            return len(str(value).encode('utf-8'))
    
    def _start_background_optimization(self) -> None:
        """Start background cache optimization."""
        async def optimization_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # Optimize every minute
                    await self._optimize_cache_layout()
                except Exception as e:
                    self.logger.error(f"Cache optimization error: {e}")
        
        self._optimization_task = asyncio.create_task(optimization_loop())
    
    async def _optimize_cache_layout(self) -> None:
        """Optimize cache layout based on access patterns."""
        current_time = time.time()
        
        # Find hot items that should be in L1
        hot_threshold = 10  # Access frequency threshold
        recent_threshold = 300  # 5 minutes
        
        for key in list(self.l2_cache.keys()) + list(self.l3_cache.keys()):
            frequency = self.access_frequency.get(key, 0)
            recency = current_time - self.access_recency.get(key, 0)
            
            # Promote hot and recent items
            if frequency >= hot_threshold and recency <= recent_threshold:
                if key in self.l2_cache:
                    value = self.l2_cache[key]
                    del self.l2_cache[key]
                    await self._promote_to_l1(key, value)
                elif key in self.l3_cache:
                    value = self.l3_cache[key]
                    del self.l3_cache[key]
                    await self._promote_to_l2(key, value)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_ratio = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": hit_ratio,
            "evictions": self.evictions,
            "total_size_mb": self.total_size / (1024 * 1024),
            "l1_size": len(self.l1_cache),
            "l2_size": len(self.l2_cache),
            "l3_size": len(self.l3_cache),
            "average_access_frequency": sum(self.access_frequency.values()) / len(self.access_frequency) if self.access_frequency else 0
        }


class IntelligentLoadBalancer:
    """Intelligent load balancing for distributed processing."""
    
    def __init__(self, scaling_mode: ScalingMode = ScalingMode.HYBRID):
        """Initialize load balancer.
        
        Args:
            scaling_mode: Scaling mode for processing
        """
        self.scaling_mode = scaling_mode
        self.logger = setup_logger(__name__)
        
        # Worker pools
        self.thread_pool = None
        self.process_pool = None
        
        # Load tracking
        self.worker_loads: Dict[str, float] = defaultdict(float)
        self.worker_performance: Dict[str, List[float]] = defaultdict(list)
        self.task_queue_sizes: Dict[str, int] = defaultdict(int)
        
        # Performance optimization
        self.optimization_history: List[PerformanceMetrics] = []
        
        self._initialize_workers()
    
    def _initialize_workers(self) -> None:
        """Initialize worker pools based on scaling mode."""
        cpu_count = mp.cpu_count()
        
        if self.scaling_mode in [ScalingMode.MULTI_THREAD, ScalingMode.HYBRID]:
            # Thread pool for I/O bound tasks
            self.thread_pool = ThreadPoolExecutor(
                max_workers=min(cpu_count * 2, 32),
                thread_name_prefix="formalization_thread"
            )
        
        if self.scaling_mode in [ScalingMode.MULTI_PROCESS, ScalingMode.HYBRID]:
            # Process pool for CPU bound tasks
            self.process_pool = ProcessPoolExecutor(
                max_workers=cpu_count,
                # mp_context=mp.get_context('spawn')  # For better cross-platform compatibility
            )
    
    async def submit_task(
        self,
        task_func: Callable,
        *args,
        task_type: str = "default",
        priority: int = 1,
        **kwargs
    ) -> Any:
        """Submit task for optimized execution.
        
        Args:
            task_func: Function to execute
            *args: Function arguments
            task_type: Type of task for optimization
            priority: Task priority (higher = more important)
            **kwargs: Function keyword arguments
            
        Returns:
            Task result
        """
        start_time = time.time()
        
        # Choose optimal execution method
        executor = self._choose_executor(task_func, task_type)
        
        try:
            if executor == "async":
                # Execute directly in async context
                if asyncio.iscoroutinefunction(task_func):
                    result = await task_func(*args, **kwargs)
                else:
                    result = task_func(*args, **kwargs)
            
            elif executor == "thread":
                # Execute in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool, 
                    lambda: task_func(*args, **kwargs)
                )
            
            elif executor == "process":
                # Execute in process pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.process_pool,
                    lambda: task_func(*args, **kwargs)
                )
            
            else:
                # Fallback to direct execution
                result = task_func(*args, **kwargs)
            
            # Record performance
            execution_time = time.time() - start_time
            self._record_task_performance(task_type, executor, execution_time, True)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_task_performance(task_type, executor, execution_time, False)
            raise e
    
    def _choose_executor(self, task_func: Callable, task_type: str) -> str:
        """Choose optimal executor for task."""
        # Task type heuristics
        if task_type in ["parsing", "generation", "computation"]:
            # CPU-intensive tasks
            if self.process_pool:
                return "process"
            elif self.thread_pool:
                return "thread"
        
        elif task_type in ["verification", "api_call", "io"]:
            # I/O-intensive tasks
            if self.thread_pool:
                return "thread"
        
        elif task_type in ["validation", "caching"]:
            # Lightweight tasks
            return "async"
        
        # Adaptive choice based on historical performance
        best_executor = self._get_best_executor_for_type(task_type)
        if best_executor:
            return best_executor
        
        # Default choice
        return "async"
    
    def _get_best_executor_for_type(self, task_type: str) -> Optional[str]:
        """Get best executor based on historical performance."""
        if not self.optimization_history:
            return None
        
        # Analyze recent performance data
        recent_metrics = self.optimization_history[-10:]  # Last 10 measurements
        executor_performance = defaultdict(list)
        
        for metrics in recent_metrics:
            # This would be expanded with actual performance tracking
            pass
        
        # Return executor with best average performance
        if executor_performance:
            best_executor = min(executor_performance.keys(),
                              key=lambda x: sum(executor_performance[x]) / len(executor_performance[x]))
            return best_executor
        
        return None
    
    def _record_task_performance(
        self,
        task_type: str,
        executor: str,
        execution_time: float,
        success: bool
    ) -> None:
        """Record task performance for optimization."""
        worker_id = f"{executor}_{task_type}"
        
        # Update load tracking
        self.worker_loads[worker_id] = execution_time
        self.worker_performance[worker_id].append(execution_time)
        
        # Keep only recent performance data
        if len(self.worker_performance[worker_id]) > 100:
            self.worker_performance[worker_id] = self.worker_performance[worker_id][-50:]
    
    async def batch_process(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        batch_size: int = 10,
        max_concurrent: int = None
    ) -> List[Any]:
        """Process tasks in optimized batches.
        
        Args:
            tasks: List of (function, args, kwargs) tuples
            batch_size: Size of processing batches
            max_concurrent: Maximum concurrent tasks
            
        Returns:
            List of results
        """
        if max_concurrent is None:
            max_concurrent = mp.cpu_count() * 2
        
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_task(task_info):
            async with semaphore:
                func, args, kwargs = task_info
                return await self.submit_task(func, *args, **kwargs)
        
        # Process in batches
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_tasks = [process_task(task_info) for task_info in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get load balancer performance statistics."""
        return {
            "worker_loads": dict(self.worker_loads),
            "average_performance": {
                worker: sum(times) / len(times)
                for worker, times in self.worker_performance.items()
                if times
            },
            "active_workers": {
                "thread_pool": self.thread_pool is not None,
                "process_pool": self.process_pool is not None
            },
            "scaling_mode": self.scaling_mode.value
        }
    
    def shutdown(self) -> None:
        """Shutdown worker pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class AdvancedOptimizationEngine:
    """Advanced optimization engine coordinating all performance enhancements."""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize optimization engine.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.logger = setup_logger(__name__)
        
        # Initialize components
        self.cache = AdaptiveCache(
            max_size_mb=self.config.cache_size_mb,
            enable_compression=self.config.enable_compression,
            enable_persistence=self.config.enable_persistent_cache
        )
        
        self.load_balancer = IntelligentLoadBalancer(
            scaling_mode=self.config.scaling_mode
        )
        
        # Performance monitoring
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_intervals: List[float] = []
        
        # Adaptive optimization
        self.current_strategy = self.config.strategy
        self.performance_targets = {
            "throughput": self.config.performance_target_rps,
            "latency": 1.0,  # seconds
            "cache_hit_ratio": 0.8,
            "error_rate": 0.05
        }
        
        # Background optimization
        self._optimization_task = None
        if self.config.strategy == OptimizationStrategy.ADAPTIVE:
            self._start_adaptive_optimization()
    
    async def optimize_operation(
        self,
        operation_func: Callable,
        cache_key: Optional[str] = None,
        task_type: str = "default",
        *args,
        **kwargs
    ) -> Any:
        """Optimize a single operation with caching and load balancing.
        
        Args:
            operation_func: Function to optimize
            cache_key: Optional cache key
            task_type: Type of task for optimization
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Operation result
        """
        start_time = time.time()
        
        # Check cache first
        if cache_key:
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result
        
        # Execute with load balancing
        try:
            result = await self.load_balancer.submit_task(
                operation_func,
                *args,
                task_type=task_type,
                **kwargs
            )
            
            # Cache the result
            if cache_key and result is not None:
                await self.cache.set(cache_key, result)
            
            # Record metrics
            execution_time = time.time() - start_time
            self._record_operation_metrics(execution_time, True, task_type)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_operation_metrics(execution_time, False, task_type)
            raise e
    
    async def optimize_batch(
        self,
        operations: List[Tuple[Callable, tuple, dict]],
        cache_keys: Optional[List[str]] = None,
        task_type: str = "batch"
    ) -> List[Any]:
        """Optimize batch of operations.
        
        Args:
            operations: List of (function, args, kwargs) tuples
            cache_keys: Optional cache keys for each operation
            task_type: Type of batch task
            
        Returns:
            List of results
        """
        start_time = time.time()
        
        # Check cache for each operation
        results = []
        uncached_operations = []
        uncached_indices = []
        
        if cache_keys:
            for i, (cache_key, (func, args, kwargs)) in enumerate(zip(cache_keys, operations)):
                if cache_key:
                    cached_result = await self.cache.get(cache_key)
                    if cached_result is not None:
                        results.append(cached_result)
                        continue
                
                results.append(None)  # Placeholder
                uncached_operations.append((func, args, kwargs))
                uncached_indices.append(i)
        else:
            results = [None] * len(operations)
            uncached_operations = operations
            uncached_indices = list(range(len(operations)))
        
        # Process uncached operations
        if uncached_operations:
            uncached_results = await self.load_balancer.batch_process(
                uncached_operations,
                batch_size=self.config.batch_size
            )
            
            # Fill in results and cache
            for idx, result, op_idx in zip(uncached_indices, uncached_results, range(len(uncached_operations))):
                results[idx] = result
                
                # Cache result if successful and cache key provided
                if (cache_keys and cache_keys[idx] and 
                    not isinstance(result, Exception) and result is not None):
                    await self.cache.set(cache_keys[idx], result)
        
        # Record batch metrics
        execution_time = time.time() - start_time
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        self._record_batch_metrics(len(operations), success_count, execution_time, task_type)
        
        return results
    
    def _record_operation_metrics(
        self,
        execution_time: float,
        success: bool,
        task_type: str
    ) -> None:
        """Record metrics for single operation."""
        cache_stats = self.cache.get_statistics()
        
        metrics = PerformanceMetrics(
            operation_count=1,
            total_time=execution_time,
            average_time=execution_time,
            cache_hits=cache_stats["hits"],
            cache_misses=cache_stats["misses"],
            cache_hit_ratio=cache_stats["hit_ratio"],
            throughput=1.0 / execution_time if execution_time > 0 else 0,
            error_rate=0.0 if success else 1.0
        )
        
        self.metrics_history.append(metrics)
        
        # Keep history bounded
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]
    
    def _record_batch_metrics(
        self,
        total_operations: int,
        successful_operations: int,
        execution_time: float,
        task_type: str
    ) -> None:
        """Record metrics for batch operation."""
        cache_stats = self.cache.get_statistics()
        
        metrics = PerformanceMetrics(
            operation_count=total_operations,
            total_time=execution_time,
            average_time=execution_time / total_operations if total_operations > 0 else 0,
            cache_hits=cache_stats["hits"],
            cache_misses=cache_stats["misses"],
            cache_hit_ratio=cache_stats["hit_ratio"],
            throughput=total_operations / execution_time if execution_time > 0 else 0,
            error_rate=(total_operations - successful_operations) / total_operations if total_operations > 0 else 0
        )
        
        self.metrics_history.append(metrics)
    
    def _start_adaptive_optimization(self) -> None:
        """Start adaptive optimization background task."""
        async def optimization_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.optimization_interval)
                    await self._perform_adaptive_optimization()
                except Exception as e:
                    self.logger.error(f"Adaptive optimization error: {e}")
        
        self._optimization_task = asyncio.create_task(optimization_loop())
    
    async def _perform_adaptive_optimization(self) -> None:
        """Perform adaptive optimization based on performance metrics."""
        if len(self.metrics_history) < 10:
            return
        
        # Analyze recent performance
        recent_metrics = self.metrics_history[-10:]
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit_ratio = sum(m.cache_hit_ratio for m in recent_metrics) / len(recent_metrics)
        
        # Determine if optimization is needed
        optimizations_applied = []
        
        # Throughput optimization
        if avg_throughput < self.performance_targets["throughput"] * 0.8:
            await self._optimize_throughput()
            optimizations_applied.append("throughput")
        
        # Error rate optimization
        if avg_error_rate > self.performance_targets["error_rate"]:
            await self._optimize_error_rate()
            optimizations_applied.append("error_rate")
        
        # Cache optimization
        if avg_cache_hit_ratio < self.performance_targets["cache_hit_ratio"]:
            await self._optimize_cache_performance()
            optimizations_applied.append("cache")
        
        if optimizations_applied:
            self.logger.info(f"Applied adaptive optimizations: {optimizations_applied}")
    
    async def _optimize_throughput(self) -> None:
        """Optimize system throughput."""
        # Increase batch size if performance allows
        if self.config.batch_size < 50:
            self.config.batch_size = min(self.config.batch_size * 2, 50)
            self.logger.info(f"Increased batch size to {self.config.batch_size}")
        
        # Adjust caching strategy
        if self.cache.max_size_bytes < 1024 * 1024 * 1024:  # 1GB limit
            self.cache.max_size_bytes = min(self.cache.max_size_bytes * 1.5, 1024 * 1024 * 1024)
            self.logger.info(f"Increased cache size to {self.cache.max_size_bytes / (1024*1024):.0f}MB")
    
    async def _optimize_error_rate(self) -> None:
        """Optimize system error rate."""
        # Reduce batch size to improve reliability
        if self.config.batch_size > 1:
            self.config.batch_size = max(self.config.batch_size // 2, 1)
            self.logger.info(f"Reduced batch size to {self.config.batch_size} for reliability")
    
    async def _optimize_cache_performance(self) -> None:
        """Optimize cache performance."""
        # Trigger cache optimization
        await self.cache._optimize_cache_layout()
        self.logger.info("Triggered cache layout optimization")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        return {
            "current_strategy": self.current_strategy.value,
            "configuration": {
                "scaling_mode": self.config.scaling_mode.value,
                "batch_size": self.config.batch_size,
                "cache_size_mb": self.config.cache_size_mb,
                "max_workers": self.config.max_workers
            },
            "performance_metrics": {
                "average_throughput": sum(m.throughput for m in recent_metrics) / len(recent_metrics),
                "average_latency": sum(m.average_time for m in recent_metrics) / len(recent_metrics),
                "average_error_rate": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
                "cache_hit_ratio": sum(m.cache_hit_ratio for m in recent_metrics) / len(recent_metrics)
            },
            "cache_statistics": self.cache.get_statistics(),
            "load_balancer_statistics": self.load_balancer.get_performance_statistics(),
            "performance_targets": self.performance_targets,
            "total_operations": sum(m.operation_count for m in self.metrics_history),
            "optimization_intervals": len(self.optimization_intervals)
        }
    
    def shutdown(self) -> None:
        """Shutdown optimization engine."""
        if self._optimization_task:
            self._optimization_task.cancel()
        
        self.load_balancer.shutdown()
        
        self.logger.info("Optimization engine shutdown complete")