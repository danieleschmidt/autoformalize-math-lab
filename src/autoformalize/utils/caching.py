"""Advanced caching system for the formalization pipeline.

This module provides intelligent caching mechanisms to improve performance
by avoiding redundant computations and API calls.
"""

import hashlib
import json
import pickle
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

try:
    import redis
    HAS_REDIS = True
except ImportError:
    redis = None
    HAS_REDIS = False

from .logging_config import setup_logger


class CacheStrategy(Enum):
    """Cache strategies for different types of data."""
    LRU = "lru"          # Least Recently Used
    TTL = "ttl"          # Time To Live
    ADAPTIVE = "adaptive" # Adaptive based on access patterns
    PERSISTENT = "persistent" # Persistent across sessions


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


class CacheManager:
    """Intelligent cache manager with multiple strategies and backends.
    
    This class provides caching for expensive operations like LLM API calls,
    LaTeX parsing results, and verification outcomes.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        redis_url: Optional[str] = None,
        max_memory_entries: int = 1000,
        default_ttl: int = 3600  # 1 hour
    ):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for persistent cache files
            redis_url: Redis connection URL for distributed caching
            max_memory_entries: Maximum entries in memory cache
            default_ttl: Default TTL in seconds
        """
        self.cache_dir = cache_dir or Path.home() / ".autoformalize" / "cache"
        self.max_memory_entries = max_memory_entries
        self.default_ttl = default_ttl
        self.logger = setup_logger(__name__)
        
        # Memory cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        # Redis cache
        self.redis_client = None
        if redis_url and HAS_REDIS:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()  # Test connection
                self.logger.info("Redis cache backend initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Redis: {e}")
        
        # File cache setup
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "errors": 0
        }
        
        # Cache strategies configuration
        self.cache_strategies = {
            "llm_responses": CacheStrategy.TTL,
            "parsing_results": CacheStrategy.PERSISTENT,
            "verification_results": CacheStrategy.ADAPTIVE,
            "template_renders": CacheStrategy.LRU,
        }
        
        # TTL configuration by data type
        self.ttl_config = {
            "llm_responses": 3600,      # 1 hour
            "parsing_results": 86400,   # 24 hours
            "verification_results": 1800, # 30 minutes
            "template_renders": 7200,   # 2 hours
        }
    
    async def get(
        self,
        key: str,
        data_type: str = "default"
    ) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            data_type: Type of data for strategy selection
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            # Try memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.is_expired:
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    self.stats["hits"] += 1
                    return entry.value
                else:
                    # Remove expired entry
                    del self.memory_cache[key]
            
            # Try Redis cache
            if self.redis_client:
                try:
                    cached_data = await self._redis_get(key)
                    if cached_data:
                        # Move to memory cache for faster access
                        await self.set(key, cached_data, data_type)
                        self.stats["hits"] += 1
                        return cached_data
                except Exception as e:
                    self.logger.warning(f"Redis get failed: {e}")
            
            # Try file cache for persistent data
            if self._should_use_file_cache(data_type):
                cached_data = await self._file_get(key)
                if cached_data:
                    # Move to memory cache
                    await self.set(key, cached_data, data_type)
                    self.stats["hits"] += 1
                    return cached_data
            
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            self.stats["errors"] += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        data_type: str = "default",
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            data_type: Type of data for strategy selection
            ttl: Time to live in seconds (overrides default)
            
        Returns:
            True if successfully cached
        """
        try:
            # Determine TTL
            if ttl is None:
                ttl = self.ttl_config.get(data_type, self.default_ttl)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl,
                metadata={"data_type": data_type}
            )
            
            # Store in memory cache
            self.memory_cache[key] = entry
            
            # Evict if over limit
            if len(self.memory_cache) > self.max_memory_entries:
                await self._evict_memory_cache()
            
            # Store in Redis if available
            if self.redis_client:
                try:
                    await self._redis_set(key, value, ttl)
                except Exception as e:
                    self.logger.warning(f"Redis set failed: {e}")
            
            # Store in file cache if persistent
            if self._should_use_file_cache(data_type):
                await self._file_set(key, value)
            
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
            self.stats["errors"] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted
        """
        try:
            deleted = False
            
            # Delete from memory
            if key in self.memory_cache:
                del self.memory_cache[key]
                deleted = True
            
            # Delete from Redis
            if self.redis_client:
                try:
                    await self._redis_delete(key)
                    deleted = True
                except Exception as e:
                    self.logger.warning(f"Redis delete failed: {e}")
            
            # Delete from file cache
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
                deleted = True
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self, data_type: Optional[str] = None) -> int:
        """Clear cache entries.
        
        Args:
            data_type: Specific data type to clear (all if None)
            
        Returns:
            Number of entries cleared
        """
        cleared = 0
        
        try:
            # Clear memory cache
            if data_type:
                keys_to_delete = [
                    key for key, entry in self.memory_cache.items()
                    if entry.metadata.get("data_type") == data_type
                ]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                    cleared += 1
            else:
                cleared = len(self.memory_cache)
                self.memory_cache.clear()
            
            # Clear Redis (if specific data type, would need pattern matching)
            if self.redis_client and not data_type:
                try:
                    await self._redis_clear()
                except Exception as e:
                    self.logger.warning(f"Redis clear failed: {e}")
            
            # Clear file cache
            if not data_type:
                for file_path in self.cache_dir.glob("*.cache"):
                    file_path.unlink()
            
            self.logger.info(f"Cleared {cleared} cache entries")
            return cleared
            
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "memory_entries": len(self.memory_cache),
            "cache_strategies": {k: v.value for k, v in self.cache_strategies.items()},
            "redis_available": self.redis_client is not None,
            "cache_dir": str(self.cache_dir)
        }
    
    def _generate_cache_key(self, operation: str, **params) -> str:
        """Generate cache key from operation and parameters.
        
        Args:
            operation: Operation name
            **params: Parameters that affect the result
            
        Returns:
            Cache key string
        """
        # Create deterministic key from operation and parameters
        key_data = {"operation": operation, "params": params}
        key_json = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_json.encode()).hexdigest()[:16]
        return f"{operation}:{key_hash}"
    
    async def _evict_memory_cache(self) -> None:
        """Evict entries from memory cache using LRU strategy."""
        if len(self.memory_cache) <= self.max_memory_entries:
            return
        
        # Sort by last accessed time (LRU)
        entries = list(self.memory_cache.items())
        entries.sort(key=lambda x: x[1].last_accessed)
        
        # Remove oldest 10% of entries
        num_to_remove = max(1, len(entries) // 10)
        
        for i in range(num_to_remove):
            key, _ = entries[i]
            del self.memory_cache[key]
            self.stats["evictions"] += 1
    
    def _should_use_file_cache(self, data_type: str) -> bool:
        """Determine if data type should use file caching."""
        strategy = self.cache_strategies.get(data_type, CacheStrategy.LRU)
        return strategy == CacheStrategy.PERSISTENT
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"
    
    async def _file_get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"File cache read error: {e}")
            return None
    
    async def _file_set(self, key: str, value: Any) -> None:
        """Set value in file cache."""
        file_path = self._get_file_path(key)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            self.logger.warning(f"File cache write error: {e}")
    
    async def _redis_get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            self.logger.warning(f"Redis get error: {e}")
        
        return None
    
    async def _redis_set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in Redis cache."""
        if not self.redis_client:
            return
        
        try:
            data = pickle.dumps(value)
            self.redis_client.setex(key, ttl, data)
        except Exception as e:
            self.logger.warning(f"Redis set error: {e}")
    
    async def _redis_delete(self, key: str) -> None:
        """Delete key from Redis cache."""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.delete(key)
        except Exception as e:
            self.logger.warning(f"Redis delete error: {e}")
    
    async def _redis_clear(self) -> None:
        """Clear all Redis cache."""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.flushdb()
        except Exception as e:
            self.logger.warning(f"Redis clear error: {e}")


def cached(
    data_type: str = "default",
    ttl: Optional[int] = None,
    cache_manager: Optional[CacheManager] = None
):
    """Decorator for caching function results.
    
    Args:
        data_type: Type of data for cache strategy
        ttl: Time to live in seconds
        cache_manager: CacheManager instance (creates default if None)
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            manager = cache_manager or global_cache_manager
            
            # Generate cache key
            key = manager._generate_cache_key(func.__name__, args=args, kwargs=kwargs)
            
            # Try to get from cache
            result = await manager.get(key, data_type)
            if result is not None:
                return result
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            await manager.set(key, result, data_type, ttl)
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            manager = cache_manager or global_cache_manager
            
            # For sync functions, use blocking cache operations
            key = manager._generate_cache_key(func.__name__, args=args, kwargs=kwargs)
            
            # Simple memory-only caching for sync functions
            if key in manager.memory_cache:
                entry = manager.memory_cache[key]
                if not entry.is_expired:
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    manager.stats["hits"] += 1
                    return entry.value
            
            # Call function and cache result
            result = func(*args, **kwargs)
            
            entry = CacheEntry(
                key=key,
                value=result,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl or manager.default_ttl,
                metadata={"data_type": data_type}
            )
            manager.memory_cache[key] = entry
            manager.stats["sets"] += 1
            
            return result
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global cache manager instance
global_cache_manager = CacheManager()