"""Optimized formalization pipeline with performance enhancements."""

import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque

from .robust_pipeline import RobustFormalizationPipeline, RobustFormalizationResult
from .pipeline import TargetSystem, FormalizationResult
from .config import FormalizationConfig
from .exceptions import FormalizationError, TimeoutError
from ..utils.logging_config import setup_logger
from ..utils.metrics import FormalizationMetrics


@dataclass
class CacheEntry:
    """Cache entry for memoized formalization results."""
    result: FormalizationResult
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    content_hash: str = ""
    size_bytes: int = 0
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class OptimizationSettings:
    """Settings for performance optimization."""
    enable_caching: bool = True
    cache_max_size: int = 1000
    cache_ttl: float = 3600.0  # 1 hour
    enable_parallel_processing: bool = True
    max_concurrent_requests: int = 10
    batch_processing_enabled: bool = True
    max_batch_size: int = 50


class IntelligentCache:
    """High-performance intelligent cache with LRU and TTL eviction."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600.0):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = deque()  # LRU tracking
        self._lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_bytes': 0
        }
    
    def _make_key(self, content: str, target_system: str, **kwargs) -> str:
        """Create cache key from content and parameters."""
        key_data = f"{content}:{target_system}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _evict_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry.timestamp > self.ttl
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
            self.stats['evictions'] += 1
    
    def _evict_lru(self) -> None:
        """Remove least recently used entries to maintain size limit."""
        while len(self._cache) >= self.max_size and self._access_order:
            lru_key = self._access_order.popleft()
            if lru_key in self._cache:  # Key might have been removed already
                self._remove_entry(lru_key)
                self.stats['evictions'] += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remove cache entry and update stats."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self.stats['size_bytes'] -= entry.size_bytes
            try:
                self._access_order.remove(key)
            except ValueError:
                pass  # Already removed
    
    def get(self, content: str, target_system: str, **kwargs) -> Optional[FormalizationResult]:
        """Get cached result if available."""
        with self._lock:
            self._evict_expired()
            
            key = self._make_key(content, target_system, **kwargs)
            entry = self._cache.get(key)
            
            if entry:
                entry.update_access()
                # Move to end of access order (most recently used)
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                self._access_order.append(key)
                
                self.stats['hits'] += 1
                return entry.result
            
            self.stats['misses'] += 1
            return None
    
    def put(self, content: str, target_system: str, result: FormalizationResult, **kwargs) -> None:
        """Store result in cache."""
        with self._lock:
            key = self._make_key(content, target_system, **kwargs)
            
            # Estimate size
            size_bytes = len(content) + len(str(result.formal_code or ""))
            
            # Create cache entry
            entry = CacheEntry(
                result=result,
                timestamp=time.time(),
                content_hash=key,
                size_bytes=size_bytes
            )
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Ensure we have space
            self._evict_lru()
            
            # Add new entry
            self._cache[key] = entry
            self._access_order.append(key)
            self.stats['size_bytes'] += size_bytes
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses'])
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'current_size': len(self._cache),
                'max_size': self.max_size,
                'fill_percentage': len(self._cache) / self.max_size * 100
            }
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self.stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'size_bytes': 0}


class OptimizedFormalizationPipeline:
    """Highly optimized formalization pipeline with advanced performance features."""
    
    def __init__(
        self,
        target_system: TargetSystem,
        config: Optional[FormalizationConfig] = None,
        optimization_settings: Optional[OptimizationSettings] = None
    ):
        self.target_system = target_system
        self.config = config or FormalizationConfig()
        self.optimization_settings = optimization_settings or OptimizationSettings()
        self.logger = setup_logger(__name__)
        
        # Initialize base pipeline
        self.base_pipeline = RobustFormalizationPipeline(
            target_system=target_system,
            config=self.config
        )
        
        # Initialize optimization components
        self._initialize_optimization_components()
        
        # Performance tracking
        self._active_requests = set()
        
        self.logger.info(f"Initialized optimized pipeline for {target_system.value}")
    
    def _initialize_optimization_components(self) -> None:
        """Initialize optimization-specific components."""
        
        # Intelligent caching
        if self.optimization_settings.enable_caching:
            self.cache = IntelligentCache(
                max_size=self.optimization_settings.cache_max_size,
                ttl=self.optimization_settings.cache_ttl
            )
        else:
            self.cache = None
        
        # Thread pools for concurrent processing
        if self.optimization_settings.enable_parallel_processing:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.optimization_settings.max_concurrent_requests,
                thread_name_prefix="formalization_"
            )
        else:
            self.thread_pool = None
    
    async def formalize_optimized(
        self,
        latex_content: str,
        use_cache: bool = True,
        **kwargs
    ) -> RobustFormalizationResult:
        """Optimized formalization with all performance enhancements."""
        
        request_id = id(latex_content)
        self._active_requests.add(request_id)
        
        try:
            # Check cache first
            if use_cache and self.cache:
                cached_result = self.cache.get(latex_content, self.target_system.value, **kwargs)
                if cached_result:
                    self.logger.debug("Cache hit for formalization request")
                    return RobustFormalizationResult(
                        success=cached_result.success,
                        formal_code=cached_result.formal_code,
                        error_message=cached_result.error_message,
                        verification_status=cached_result.verification_status,
                        metrics=cached_result.metrics,
                        correction_rounds=cached_result.correction_rounds,
                        processing_time=0.001,  # Cache hit time
                        warnings=["Result from cache"]
                    )
            
            # Execute formalization
            result = await self.base_pipeline.formalize_robust(
                latex_content=latex_content,
                **kwargs
            )
            
            # Store in cache if successful
            if use_cache and self.cache and result.success:
                self.cache.put(latex_content, self.target_system.value, result, **kwargs)
            
            return result
            
        finally:
            self._active_requests.discard(request_id)
    
    async def formalize_batch(
        self,
        latex_contents: List[str],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[RobustFormalizationResult]:
        """Process multiple formalizations in optimized batches."""
        
        if not self.optimization_settings.batch_processing_enabled:
            # Process sequentially
            results = []
            for content in latex_contents:
                result = await self.formalize_optimized(content, **kwargs)
                results.append(result)
            return results
        
        batch_size = batch_size or self.optimization_settings.max_batch_size
        results = []
        
        # Process in batches
        for i in range(0, len(latex_contents), batch_size):
            batch = latex_contents[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.formalize_optimized(content, **kwargs)
                for content in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions in batch results
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append(RobustFormalizationResult(
                        success=False,
                        error_message=str(result),
                        processing_time=0.0
                    ))
                else:
                    results.append(result)
        
        return results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            'target_system': self.target_system.value,
            'active_requests': len(self._active_requests),
            'optimization_settings': {
                'caching_enabled': self.optimization_settings.enable_caching,
                'parallel_processing_enabled': self.optimization_settings.enable_parallel_processing,
                'max_concurrent_requests': self.optimization_settings.max_concurrent_requests,
                'batch_processing_enabled': self.optimization_settings.batch_processing_enabled
            }
        }
        
        # Add cache statistics
        if self.cache:
            stats['cache'] = self.cache.get_stats()
        
        return stats
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Run performance optimization analysis and suggestions."""
        stats = self.get_optimization_stats()
        suggestions = []
        
        # Cache optimization suggestions
        if self.cache:
            cache_stats = stats.get('cache', {})
            hit_rate = cache_stats.get('hit_rate', 0)
            if hit_rate < 0.3:
                suggestions.append("Low cache hit rate. Consider increasing cache size or TTL.")
            elif hit_rate > 0.9 and cache_stats.get('fill_percentage', 0) < 50:
                suggestions.append("Excellent cache hit rate but low utilization. Consider reducing cache size.")
        
        return {
            'current_stats': stats,
            'optimization_suggestions': suggestions,
            'timestamp': time.time()
        }
    
    async def shutdown_gracefully(self) -> None:
        """Gracefully shutdown the optimized pipeline."""
        self.logger.info("Shutting down optimized pipeline")
        
        try:
            # Wait for active requests to complete (with timeout)
            timeout = 30.0
            start_time = time.time()
            
            while self._active_requests and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)
            
            # Shutdown thread pools
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            
            # Clear cache
            if self.cache:
                self.cache.clear()
            
            self.logger.info("Optimized pipeline shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during optimized pipeline shutdown: {e}")