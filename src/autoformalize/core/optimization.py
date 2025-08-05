"""Performance optimization and caching system.

This module provides performance optimization features including
caching, concurrent processing, and adaptive resource management.
"""

import asyncio
import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from abc import ABC, abstractmethod

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    redis = None

from ..utils.logging_config import setup_logger
from ..utils.metrics import FormalizationMetrics

T = TypeVar('T')


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass 
class PerformanceProfile:
    """Performance profile for optimization decisions."""
    avg_processing_time: float = 0.0
    peak_memory_usage: int = 0
    concurrent_requests: int = 0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "avg_processing_time": self.avg_processing_time,
            "peak_memory_usage": self.peak_memory_usage,
            "concurrent_requests": self.concurrent_requests,
            "cache_hit_rate": self.cache_hit_rate,
            "error_rate": self.error_rate
        }


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """In-memory LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = 3600):
        """Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict = OrderedDict()
        self._expiry: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            # Check if key exists and not expired
            if key in self._cache:
                if key in self._expiry and time.time() > self._expiry[key]:
                    # Expired
                    del self._cache[key]
                    del self._expiry[key]
                    self._stats.misses += 1
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._stats.hits += 1
                return self._cache[key]
            
            self._stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self._lock:
            # Remove if already exists
            if key in self._cache:
                del self._cache[key]
                if key in self._expiry:
                    del self._expiry[key]
            
            # Add new entry
            self._cache[key] = value
            
            # Set expiry
            if ttl is not None or self.default_ttl is not None:
                expiry_time = time.time() + (ttl or self.default_ttl)
                self._expiry[key] = expiry_time
            
            # Evict if over capacity
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                if oldest_key in self._expiry:
                    del self._expiry[oldest_key]
                self._stats.evictions += 1
            
            self._stats.total_size = len(self._cache)
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._expiry:
                    del self._expiry[key]
                self._stats.total_size = len(self._cache)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
            self._stats.total_size = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.total_size = len(self._cache)
            return self._stats


class RedisCache(CacheBackend):
    """Redis-based cache implementation."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "autoformalize:"):
        """Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for namespacing
        """
        if not HAS_REDIS:
            raise ImportError("Redis package required for RedisCache")
        
        self.redis_url = redis_url
        self.prefix = prefix
        self._stats = CacheStats()
        self._redis = None
        
    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            r = await self._get_redis()
            prefixed_key = self._make_key(key)
            
            data = r.get(prefixed_key)
            if data is not None:
                self._stats.hits += 1
                return pickle.loads(data.encode('latin1'))
            
            self._stats.misses += 1
            return None
            
        except Exception:
            self._stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        try:
            r = await self._get_redis()
            prefixed_key = self._make_key(key)
            
            serialized = pickle.dumps(value).decode('latin1')
            
            if ttl is not None:
                r.setex(prefixed_key, ttl, serialized)
            else:
                r.set(prefixed_key, serialized)
                
        except Exception:
            pass  # Silently fail for cache errors
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            r = await self._get_redis()
            prefixed_key = self._make_key(key)
            return bool(r.delete(prefixed_key))
        except Exception:
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            r = await self._get_redis()
            pattern = f"{self.prefix}*"
            keys = r.keys(pattern)
            if keys:
                r.delete(*keys)
        except Exception:
            pass
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class AdaptiveCache:
    """Adaptive caching system that chooses optimal caching strategies."""
    
    def __init__(
        self,
        memory_cache_size: int = 1000,
        redis_url: Optional[str] = None,
        enable_adaptive: bool = True
    ):
        """Initialize adaptive cache.
        
        Args:
            memory_cache_size: Size of memory cache
            redis_url: Redis URL (if None, only memory cache used)
            enable_adaptive: Whether to enable adaptive behavior
        """
        self.enable_adaptive = enable_adaptive
        self.logger = setup_logger(__name__)
        
        # Initialize cache backends
        self.memory_cache = MemoryCache(max_size=memory_cache_size)
        
        self.redis_cache = None
        if redis_url and HAS_REDIS:
            try:
                self.redis_cache = RedisCache(redis_url)
            except Exception as e:
                self.logger.warning(f"Failed to initialize Redis cache: {e}")
        
        # Performance tracking
        self._access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._size_stats: Dict[str, int] = {}
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with adaptive strategy."""
        access_time = time.time()
        
        # Try memory cache first (fastest)
        value = await self.memory_cache.get(key)
        if value is not None:
            self._record_access(key, access_time, "memory_hit")
            return value
        
        # Try Redis cache if available
        if self.redis_cache is not None:
            value = await self.redis_cache.get(key)
            if value is not None:
                # Store in memory cache for future access
                await self.memory_cache.set(key, value, ttl=300)  # 5 min TTL
                self._record_access(key, access_time, "redis_hit")
                return value
        
        self._record_access(key, access_time, "miss")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with adaptive strategy."""
        # Estimate value size
        try:
            size = len(pickle.dumps(value))
            self._size_stats[key] = size
        except Exception:
            size = 1000  # Default estimate
        
        # Decide caching strategy based on size and access patterns
        strategy = self._choose_caching_strategy(key, size)
        
        if strategy == "memory_only" or strategy == "both":
            await self.memory_cache.set(key, value, ttl)
        
        if strategy == "redis_only" or strategy == "both":
            if self.redis_cache is not None:
                await self.redis_cache.set(key, value, ttl)
    
    def _choose_caching_strategy(self, key: str, size: int) -> str:
        """Choose optimal caching strategy.
        
        Args:
            key: Cache key
            size: Estimated size of value
            
        Returns:
            Strategy: "memory_only", "redis_only", or "both"
        """
        if not self.enable_adaptive:
            return "both" if self.redis_cache else "memory_only"
        
        # Large objects go to Redis
        if size > 100000:  # 100KB
            return "redis_only" if self.redis_cache else "memory_only"
        
        # Frequently accessed items go to memory
        if key in self._access_patterns:
            recent_accesses = [
                t for t in self._access_patterns[key] 
                if time.time() - t < 3600  # Last hour
            ]
            if len(recent_accesses) > 10:  # Frequently accessed
                return "both" if self.redis_cache else "memory_only"
        
        # Default strategy
        return "memory_only"
    
    def _record_access(self, key: str, access_time: float, result: str) -> None:
        """Record cache access for adaptive learning."""
        self._access_patterns[key].append(access_time)
        
        # Keep only recent accesses (last 24 hours)
        cutoff = access_time - 86400
        self._access_patterns[key] = [
            t for t in self._access_patterns[key] if t > cutoff
        ]
    
    async def delete(self, key: str) -> bool:
        """Delete from all cache backends."""
        results = []
        results.append(await self.memory_cache.delete(key))
        
        if self.redis_cache is not None:
            results.append(await self.redis_cache.delete(key))
        
        # Clean up tracking data
        if key in self._access_patterns:
            del self._access_patterns[key]
        if key in self._size_stats:
            del self._size_stats[key]
        
        return any(results)
    
    async def clear(self) -> None:
        """Clear all caches."""
        await self.memory_cache.clear()
        if self.redis_cache is not None:
            await self.redis_cache.clear()
        
        self._access_patterns.clear()
        self._size_stats.clear()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        memory_stats = self.memory_cache.get_stats()
        
        stats = {
            "memory_cache": {
                "hits": memory_stats.hits,
                "misses": memory_stats.misses,
                "hit_rate": memory_stats.hit_rate,
                "size": memory_stats.total_size,
                "evictions": memory_stats.evictions
            },
            "adaptive_stats": {
                "tracked_keys": len(self._access_patterns),
                "total_accesses": sum(len(accesses) for accesses in self._access_patterns.values())
            }
        }
        
        if self.redis_cache is not None:
            redis_stats = self.redis_cache.get_stats()
            stats["redis_cache"] = {
                "hits": redis_stats.hits,
                "misses": redis_stats.misses,
                "hit_rate": redis_stats.hit_rate
            }
        
        return stats


class ResourceManager:
    """Manages computational resources and load balancing."""
    
    def __init__(
        self,
        max_concurrent_requests: int = 10,
        max_workers: int = 4,
        enable_auto_scaling: bool = True
    ):
        """Initialize resource manager.
        
        Args:
            max_concurrent_requests: Maximum concurrent requests
            max_workers: Maximum worker threads/processes
            enable_auto_scaling: Whether to enable auto-scaling
        """
        self.max_concurrent_requests = max_concurrent_requests
        self.max_workers = max_workers
        self.enable_auto_scaling = enable_auto_scaling
        self.logger = setup_logger(__name__)
        
        # Resource tracking
        self._active_requests = 0
        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._performance_history: List[PerformanceProfile] = []
        
        # Thread and process pools
        self._thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self._process_executor = ProcessPoolExecutor(max_workers=max_workers)
        
        # Resource locks
        self._resource_lock = asyncio.Semaphore(max_concurrent_requests)
        
    async def acquire_resources(self) -> None:
        """Acquire resources for processing."""
        await self._resource_lock.acquire()
        self._active_requests += 1
        
        # Auto-scaling logic
        if self.enable_auto_scaling:
            await self._check_auto_scaling()
    
    def release_resources(self) -> None:
        """Release processing resources."""
        if self._active_requests > 0:
            self._active_requests -= 1
        self._resource_lock.release()
    
    async def _check_auto_scaling(self) -> None:
        """Check if auto-scaling is needed."""
        load_factor = self._active_requests / self.max_concurrent_requests
        
        if load_factor > 0.8:  # High load
            new_limit = min(self.max_concurrent_requests * 2, 50)
            if new_limit > self.max_concurrent_requests:
                self.logger.info(f"Scaling up: {self.max_concurrent_requests} -> {new_limit}")
                self.max_concurrent_requests = new_limit
                # Create new semaphore with higher limit
                self._resource_lock = asyncio.Semaphore(new_limit)
        
        elif load_factor < 0.3:  # Low load
            new_limit = max(self.max_concurrent_requests // 2, 2)
            if new_limit < self.max_concurrent_requests:
                self.logger.info(f"Scaling down: {self.max_concurrent_requests} -> {new_limit}")
                self.max_concurrent_requests = new_limit
    
    def get_thread_executor(self) -> ThreadPoolExecutor:
        """Get thread executor for I/O-bound tasks."""
        return self._thread_executor
    
    def get_process_executor(self) -> ProcessPoolExecutor:
        """Get process executor for CPU-bound tasks."""
        return self._process_executor
    
    def record_performance(self, profile: PerformanceProfile) -> None:
        """Record performance metrics for optimization."""
        self._performance_history.append(profile)
        
        # Keep only recent history (last 100 requests)
        if len(self._performance_history) > 100:
            self._performance_history = self._performance_history[-100:]
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource utilization statistics."""
        avg_profile = self._calculate_average_performance()
        
        return {
            "active_requests": self._active_requests,
            "max_concurrent_requests": self.max_concurrent_requests,
            "utilization": self._active_requests / self.max_concurrent_requests,
            "thread_pool_size": self._thread_executor._max_workers,
            "process_pool_size": self._process_executor._max_workers,
            "performance_profile": avg_profile.to_dict() if avg_profile else None,
            "auto_scaling_enabled": self.enable_auto_scaling
        }
    
    def _calculate_average_performance(self) -> Optional[PerformanceProfile]:
        """Calculate average performance profile."""
        if not self._performance_history:
            return None
        
        avg_processing_time = sum(p.avg_processing_time for p in self._performance_history) / len(self._performance_history)
        peak_memory = max(p.peak_memory_usage for p in self._performance_history)
        avg_concurrent = sum(p.concurrent_requests for p in self._performance_history) / len(self._performance_history)
        avg_cache_hit_rate = sum(p.cache_hit_rate for p in self._performance_history) / len(self._performance_history)
        avg_error_rate = sum(p.error_rate for p in self._performance_history) / len(self._performance_history)
        
        return PerformanceProfile(
            avg_processing_time=avg_processing_time,
            peak_memory_usage=int(peak_memory),
            concurrent_requests=int(avg_concurrent),
            cache_hit_rate=avg_cache_hit_rate,
            error_rate=avg_error_rate
        )
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self._thread_executor.shutdown(wait=True)
        self._process_executor.shutdown(wait=True)


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    # Create a hash of arguments
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    
    key_string = pickle.dumps(key_data)
    return hashlib.sha256(key_string).hexdigest()


def cached(ttl: int = 3600, key_func: Optional[Callable] = None):
    """Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_func: Function to generate cache key (default uses all args)
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache (assumes cache is available globally)
            # In real implementation, cache would be injected or configured
            
            # Execute function if not cached
            result = await func(*args, **kwargs)
            
            # Store result in cache
            # await cache.set(key, result, ttl)
            
            return result
        
        return wrapper
    return decorator