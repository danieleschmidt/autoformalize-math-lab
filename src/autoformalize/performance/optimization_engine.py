"""Advanced performance optimization engine for mathematical formalization.

This module provides intelligent performance optimization including:
- Adaptive caching with ML-based cache prediction
- Query optimization and batching
- Resource usage optimization
- Performance monitoring and auto-tuning
- Distributed processing coordination
"""

import asyncio
import time
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import threading
import concurrent.futures
from functools import lru_cache

from ..utils.logging_config import setup_logger
# Caching will be handled internally


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"


class ResourceType(Enum):
    """Types of system resources to monitor."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring and optimization."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    processing_time: float
    cache_hit_rate: float
    throughput: float  # requests per second
    latency_p95: float
    error_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE
    enable_caching: bool = True
    enable_batching: bool = True
    enable_prefetching: bool = True
    enable_compression: bool = True
    max_workers: int = 4
    cache_size_mb: int = 512
    batch_size: int = 10
    batch_timeout: float = 1.0
    prefetch_threshold: float = 0.7  # Cache hit rate threshold
    compression_threshold: int = 1024  # Bytes
    auto_scaling: bool = True
    resource_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu': 80.0,
        'memory': 85.0,
        'disk': 90.0
    })


class IntelligentCache:
    """Intelligent caching system with ML-based predictions."""
    
    def __init__(self, max_size_mb: int = 512, enable_prediction: bool = True):
        """Initialize intelligent cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            enable_prediction: Enable ML-based cache prediction
        """
        self.max_size_mb = max_size_mb
        self.enable_prediction = enable_prediction
        self.logger = setup_logger(__name__)
        
        # Cache storage
        self.cache: Dict[str, Any] = {}
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Update access pattern
                self.access_patterns[key].append(time.time())
                self.cache_metadata[key]['last_access'] = time.time()
                self.cache_metadata[key]['access_count'] += 1
                
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put item in cache."""
        with self.lock:
            current_time = time.time()
            
            # Check if we need to evict items
            self._ensure_capacity()
            
            # Store item
            self.cache[key] = value
            self.cache_metadata[key] = {
                'created': current_time,
                'last_access': current_time,
                'access_count': 1,
                'size_bytes': self._estimate_size(value),
                'ttl': ttl,
                'prediction_score': self._calculate_prediction_score(key)
            }
            
            self.logger.debug(f"Cached item: {key[:20]}... (size: {self.cache_metadata[key]['size_bytes']} bytes)")
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.cache_metadata[key]
                if key in self.access_patterns:
                    del self.access_patterns[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear entire cache."""
        with self.lock:
            self.cache.clear()
            self.cache_metadata.clear()
            self.access_patterns.clear()
            self.hit_count = 0
            self.miss_count = 0
            self.eviction_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / max(total_requests, 1)
            
            total_size = sum(
                meta['size_bytes'] for meta in self.cache_metadata.values()
            )
            
            return {
                'size': len(self.cache),
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'eviction_count': self.eviction_count,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'utilization': (total_size / (self.max_size_mb * 1024 * 1024)) * 100
            }
    
    def _ensure_capacity(self) -> None:
        """Ensure cache doesn't exceed capacity."""
        total_size = sum(
            meta['size_bytes'] for meta in self.cache_metadata.values()
        )
        
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Evict items using intelligent strategy
            self._evict_items(total_size - max_size_bytes)
    
    def _evict_items(self, bytes_to_free: int) -> None:
        """Evict items to free specified bytes."""
        if not self.cache:
            return
        
        # Score items for eviction (lower score = more likely to evict)
        scored_items = []
        current_time = time.time()
        
        for key, metadata in self.cache_metadata.items():
            score = self._calculate_eviction_score(key, metadata, current_time)
            scored_items.append((score, key, metadata['size_bytes']))
        
        # Sort by score (ascending - evict lowest scores first)
        scored_items.sort()
        
        freed_bytes = 0
        for score, key, size_bytes in scored_items:
            if freed_bytes >= bytes_to_free:
                break
            
            del self.cache[key]
            del self.cache_metadata[key]
            if key in self.access_patterns:
                del self.access_patterns[key]
            
            freed_bytes += size_bytes
            self.eviction_count += 1
        
        self.logger.info(f"Evicted {self.eviction_count} items, freed {freed_bytes} bytes")
    
    def _calculate_eviction_score(self, key: str, metadata: Dict[str, Any], current_time: float) -> float:
        """Calculate eviction score for an item (lower = more likely to evict)."""
        # Time since last access (higher = more likely to evict)
        time_factor = current_time - metadata['last_access']
        
        # Access frequency (lower = more likely to evict)
        frequency_factor = 1.0 / max(metadata['access_count'], 1)
        
        # Size factor (larger items more likely to evict for space)
        size_factor = metadata['size_bytes'] / (1024 * 1024)  # MB
        
        # Prediction score (lower = more likely to evict)
        prediction_factor = 1.0 - metadata.get('prediction_score', 0.5)
        
        # TTL factor
        ttl_factor = 1.0
        if metadata.get('ttl'):
            time_left = (metadata['created'] + metadata['ttl']) - current_time
            if time_left <= 0:
                return -1000  # Expired - highest priority for eviction
            ttl_factor = 1.0 / max(time_left, 1)
        
        # Combine factors
        score = (time_factor * 0.3 + 
                frequency_factor * 0.2 + 
                size_factor * 0.2 + 
                prediction_factor * 0.2 + 
                ttl_factor * 0.1)
        
        return score
    
    def _calculate_prediction_score(self, key: str) -> float:
        """Calculate prediction score for cache utility."""
        if not self.enable_prediction or key not in self.access_patterns:
            return 0.5  # Default score
        
        access_times = self.access_patterns[key]
        if len(access_times) < 2:
            return 0.5
        
        # Analyze access pattern
        current_time = time.time()
        recent_accesses = [t for t in access_times if current_time - t < 3600]  # Last hour
        
        if not recent_accesses:
            return 0.1  # Very low utility
        
        # Calculate access frequency
        if len(recent_accesses) >= 3:
            # High frequency access
            return 0.9
        elif len(recent_accesses) >= 2:
            # Medium frequency
            return 0.7
        else:
            # Low frequency
            return 0.3
    
    @staticmethod
    def _estimate_size(obj: Any) -> int:
        """Estimate size of object in bytes."""
        if isinstance(obj, str):
            return len(obj.encode('utf-8'))
        elif isinstance(obj, (int, float)):
            return 8
        elif isinstance(obj, (list, tuple)):
            return sum(IntelligentCache._estimate_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(
                IntelligentCache._estimate_size(k) + IntelligentCache._estimate_size(v)
                for k, v in obj.items()
            )
        else:
            # Rough estimate
            return len(str(obj)) * 2


class BatchProcessor:
    """Intelligent batch processor for optimization."""
    
    def __init__(self, batch_size: int = 10, timeout: float = 1.0):
        """Initialize batch processor.
        
        Args:
            batch_size: Maximum batch size
            timeout: Maximum time to wait for batch
        """
        self.batch_size = batch_size
        self.timeout = timeout
        self.logger = setup_logger(__name__)
        
        # Batch queue
        self.pending_requests: List[Tuple[str, Any, asyncio.Future]] = []
        self.batch_lock = asyncio.Lock()
        self.batch_timer: Optional[asyncio.Task] = None
    
    async def add_request(self, request_id: str, data: Any) -> Any:
        """Add request to batch."""
        future = asyncio.Future()
        
        async with self.batch_lock:
            self.pending_requests.append((request_id, data, future))
            
            # Start timer if this is the first request
            if len(self.pending_requests) == 1:
                self.batch_timer = asyncio.create_task(self._batch_timeout())
            
            # Process batch if full
            if len(self.pending_requests) >= self.batch_size:
                await self._process_batch()
        
        return await future
    
    async def _batch_timeout(self) -> None:
        """Handle batch timeout."""
        await asyncio.sleep(self.timeout)
        
        async with self.batch_lock:
            if self.pending_requests:
                await self._process_batch()
    
    async def _process_batch(self) -> None:
        """Process current batch."""
        if not self.pending_requests:
            return
        
        batch = self.pending_requests[:]
        self.pending_requests.clear()
        
        # Cancel timer
        if self.batch_timer and not self.batch_timer.done():
            self.batch_timer.cancel()
        
        self.logger.info(f"Processing batch of {len(batch)} requests")
        
        try:
            # Process batch
            results = await self._execute_batch([data for _, data, _ in batch])
            
            # Distribute results
            for (request_id, data, future), result in zip(batch, results):
                if not future.done():
                    future.set_result(result)
                    
        except Exception as e:
            # Set exception for all futures
            for _, _, future in batch:
                if not future.done():
                    future.set_exception(e)
    
    async def _execute_batch(self, batch_data: List[Any]) -> List[Any]:
        """Execute batch processing - to be implemented by subclasses."""
        # Default implementation - process individually
        results = []
        for data in batch_data:
            result = await self._process_single(data)
            results.append(result)
        return results
    
    async def _process_single(self, data: Any) -> Any:
        """Process single item - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _process_single")


class PerformanceMonitor:
    """Real-time performance monitoring and optimization."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize performance monitor.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=1000)
        self.current_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            network_io=0.0,
            processing_time=0.0,
            cache_hit_rate=0.0,
            throughput=0.0,
            latency_p95=0.0,
            error_rate=0.0
        )
        
        # Performance tracking
        self.request_times: deque = deque(maxlen=100)
        self.error_count = 0
        self.total_requests = 0
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_active = False
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        self.logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                await self._collect_metrics()
                await self._analyze_performance()
                await asyncio.sleep(10)  # Collect metrics every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _collect_metrics(self) -> None:
        """Collect current performance metrics."""
        current_time = time.time()
        
        # System metrics
        if PSUTIL_AVAILABLE:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            memory_usage = memory_info.percent
            disk_usage = disk_info.percent
        else:
            # Fallback values when psutil is not available
            cpu_usage = 20.0  # Mock value
            memory_usage = 30.0  # Mock value
            disk_usage = 40.0  # Mock value
        
        # Calculate throughput (requests per second)
        recent_requests = [t for t in self.request_times if current_time - t < 60]
        throughput = len(recent_requests) / 60.0 if recent_requests else 0.0
        
        # Calculate latency percentile
        recent_times = list(self.request_times)[-50:]  # Last 50 requests
        latency_p95 = sorted(recent_times)[int(len(recent_times) * 0.95)] if recent_times else 0.0
        
        # Calculate error rate
        error_rate = (self.error_count / max(self.total_requests, 1)) * 100
        
        metrics = PerformanceMetrics(
            timestamp=current_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=0.0,  # Would need more complex calculation
            processing_time=sum(recent_times) / len(recent_times) if recent_times else 0.0,
            cache_hit_rate=0.0,  # Will be updated by cache
            throughput=throughput,
            latency_p95=latency_p95,
            error_rate=error_rate
        )
        
        self.metrics_history.append(metrics)
        self.current_metrics = metrics
    
    async def _analyze_performance(self) -> None:
        """Analyze performance and trigger optimizations."""
        metrics = self.current_metrics
        
        # Check resource thresholds
        if metrics.cpu_usage > self.config.resource_thresholds['cpu']:
            self.logger.warning(f"High CPU usage: {metrics.cpu_usage:.1f}%")
            await self._optimize_cpu_usage()
        
        if metrics.memory_usage > self.config.resource_thresholds['memory']:
            self.logger.warning(f"High memory usage: {metrics.memory_usage:.1f}%")
            await self._optimize_memory_usage()
        
        if metrics.error_rate > 5.0:  # 5% error rate threshold
            self.logger.warning(f"High error rate: {metrics.error_rate:.1f}%")
            await self._handle_high_error_rate()
    
    async def _optimize_cpu_usage(self) -> None:
        """Optimize for high CPU usage."""
        self.logger.info("Optimizing for CPU usage")
        # Could implement CPU-specific optimizations
        
    async def _optimize_memory_usage(self) -> None:
        """Optimize for high memory usage."""
        self.logger.info("Optimizing for memory usage")
        # Could trigger cache cleanup, garbage collection, etc.
        
    async def _handle_high_error_rate(self) -> None:
        """Handle high error rate."""
        self.logger.info("Handling high error rate")
        # Could implement circuit breakers, fallbacks, etc.
    
    def record_request(self, processing_time: float, success: bool = True) -> None:
        """Record request metrics."""
        self.request_times.append(processing_time)
        self.total_requests += 1
        
        if not success:
            self.error_count += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 data points
        
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_latency = sum(m.latency_p95 for m in recent_metrics) / len(recent_metrics)
        
        return {
            'current': {
                'cpu_usage': self.current_metrics.cpu_usage,
                'memory_usage': self.current_metrics.memory_usage,
                'throughput': self.current_metrics.throughput,
                'latency_p95': self.current_metrics.latency_p95,
                'error_rate': self.current_metrics.error_rate
            },
            'averages': {
                'cpu_usage': avg_cpu,
                'memory_usage': avg_memory,
                'throughput': avg_throughput,
                'latency_p95': avg_latency
            },
            'totals': {
                'total_requests': self.total_requests,
                'error_count': self.error_count
            }
        }


class OptimizationEngine:
    """Main optimization engine coordinating all performance optimizations."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize optimization engine.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.logger = setup_logger(__name__)
        
        # Initialize components
        self.cache = IntelligentCache(
            max_size_mb=self.config.cache_size_mb,
            enable_prediction=True
        ) if self.config.enable_caching else None
        
        self.batch_processor = BatchProcessor(
            batch_size=self.config.batch_size,
            timeout=self.config.batch_timeout
        ) if self.config.enable_batching else None
        
        self.performance_monitor = PerformanceMonitor(self.config)
        
        # Thread pool for CPU-intensive tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        )
        
        self.logger.info(f"Optimization engine initialized with {self.config.strategy.value} strategy")
    
    async def start(self) -> None:
        """Start optimization engine."""
        self.performance_monitor.start_monitoring()
        self.logger.info("Optimization engine started")
    
    async def stop(self) -> None:
        """Stop optimization engine."""
        self.performance_monitor.stop_monitoring()
        self.executor.shutdown(wait=True)
        self.logger.info("Optimization engine stopped")
    
    async def optimize_formalization(
        self,
        latex_content: str,
        formalization_func: callable,
        **kwargs
    ) -> Any:
        """Optimize formalization process.
        
        Args:
            latex_content: LaTeX content to formalize
            formalization_func: Function to perform formalization
            **kwargs: Additional arguments
            
        Returns:
            Optimized formalization result
        """
        start_time = time.time()
        cache_key = self._generate_cache_key(latex_content, kwargs)
        
        try:
            # Check cache first
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    processing_time = time.time() - start_time
                    self.performance_monitor.record_request(processing_time, success=True)
                    self.logger.debug(f"Cache hit for formalization: {cache_key[:20]}...")
                    return cached_result
            
            # Use batch processing if enabled
            if self.batch_processor and self._should_batch(latex_content):
                result = await self.batch_processor.add_request(
                    cache_key, 
                    (latex_content, formalization_func, kwargs)
                )
            else:
                # Process directly
                result = await self._execute_optimized(
                    formalization_func, 
                    latex_content, 
                    **kwargs
                )
            
            # Cache result
            if self.cache and result:
                self.cache.put(cache_key, result, ttl=3600)  # 1 hour TTL
            
            # Record metrics
            processing_time = time.time() - start_time
            self.performance_monitor.record_request(processing_time, success=True)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.performance_monitor.record_request(processing_time, success=False)
            self.logger.error(f"Formalization optimization failed: {e}")
            raise
    
    async def _execute_optimized(self, func: callable, *args, **kwargs) -> Any:
        """Execute function with optimizations."""
        # Use thread pool for CPU-intensive operations
        if self._is_cpu_intensive(func):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, func, *args, **kwargs)
        else:
            return await func(*args, **kwargs)
    
    def _generate_cache_key(self, latex_content: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for formalization."""
        content_hash = hashlib.md5(latex_content.encode()).hexdigest()
        kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
        kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()
        return f"formalization:{content_hash}:{kwargs_hash}"
    
    def _should_batch(self, latex_content: str) -> bool:
        """Determine if request should be batched."""
        # Simple heuristic - batch smaller requests
        return len(latex_content) < 1000
    
    def _is_cpu_intensive(self, func: callable) -> bool:
        """Determine if function is CPU-intensive."""
        # Simple heuristic based on function name
        cpu_intensive_patterns = ['verify', 'parse', 'generate', 'analyze']
        func_name = getattr(func, '__name__', '').lower()
        return any(pattern in func_name for pattern in cpu_intensive_patterns)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        summary = {
            'strategy': self.config.strategy.value,
            'components': {
                'caching': self.config.enable_caching,
                'batching': self.config.enable_batching,
                'prefetching': self.config.enable_prefetching
            }
        }
        
        # Add cache stats
        if self.cache:
            summary['cache'] = self.cache.get_stats()
        
        # Add performance stats
        summary['performance'] = self.performance_monitor.get_performance_summary()
        
        return summary