"""Intelligent caching system for formalization results."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import pickle

from .logging_config import setup_logger


class CacheStrategy(Enum):
    """Available caching strategies."""
    MEMORY = "memory"
    DISK = "disk"


@dataclass
class CacheEntry:
    """Represents a single cache entry."""
    key: str
    value: Any
    timestamp: float
    ttl: int
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0


class MemoryCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, max_memory_bytes: int = 100 * 1024 * 1024):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_bytes
        self.cache: Dict[str, CacheEntry] = {}
        self.logger = setup_logger(f"{__name__}.MemoryCache")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key in self.cache:
            entry = self.cache[key]
            
            # Check TTL
            if time.time() - entry.timestamp > entry.ttl:
                del self.cache[key]
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_access = time.time()
            
            return entry.value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in memory cache."""
        try:
            # Simple eviction if cache is full
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k].last_access, default=None)
                if oldest_key:
                    del self.cache[oldest_key]
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=len(str(value))  # Approximate size
            )
            
            self.cache[key] = entry
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set cache entry: {e}")
            return False
    
    async def clear(self):
        """Clear all entries from memory cache."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self.cache),
            "max_size": self.max_size,
        }


class DiskCache:
    """Simple file-based disk cache."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(f"{__name__}.DiskCache")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    entry_data = pickle.load(f)
                
                # Check TTL
                if time.time() - entry_data['timestamp'] > entry_data['ttl']:
                    cache_file.unlink()
                    return None
                
                return entry_data['value']
                
        except Exception as e:
            self.logger.error(f"Failed to get from disk cache: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in disk cache."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            entry_data = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl,
                'key': key
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(entry_data, f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set disk cache entry: {e}")
            return False
    
    async def clear(self):
        """Clear disk cache."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
        except Exception as e:
            self.logger.error(f"Failed to clear disk cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get disk cache statistics."""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            return {
                "entries": len(cache_files),
                "cache_dir": str(self.cache_dir),
            }
        except Exception as e:
            return {"entries": 0, "error": str(e)}


class CacheManager:
    """Multi-level cache manager."""
    
    def __init__(
        self,
        strategies: List[CacheStrategy] = None,
        ttl: int = 3600,
        max_memory_size: int = 100 * 1024 * 1024,
        disk_cache_dir: Path = None
    ):
        self.strategies = strategies or [CacheStrategy.MEMORY]
        self.ttl = ttl
        self.logger = setup_logger(f"{__name__}.CacheManager")
        
        # Initialize cache layers
        self.caches = {}
        
        if CacheStrategy.MEMORY in self.strategies:
            self.caches[CacheStrategy.MEMORY] = MemoryCache(
                max_memory_bytes=max_memory_size
            )
        
        if CacheStrategy.DISK in self.strategies:
            cache_dir = disk_cache_dir or Path("./cache")
            self.caches[CacheStrategy.DISK] = DiskCache(cache_dir)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy."""
        for strategy in self.strategies:
            if strategy in self.caches:
                value = await self.caches[strategy].get(key)
                if value is not None:
                    return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in all cache levels."""
        ttl = ttl or self.ttl
        success = True
        
        for strategy in self.strategies:
            if strategy in self.caches:
                result = await self.caches[strategy].set(key, value, ttl)
                success = success and result
        
        return success
    
    async def clear_all(self):
        """Clear all cache levels."""
        for cache in self.caches.values():
            await cache.clear()
    
    async def cleanup(self):
        """Cleanup cache resources."""
        pass  # Placeholder for cleanup logic
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get statistics from all cache levels."""
        stats = {}
        
        for strategy, cache in self.caches.items():
            stats[strategy.value] = cache.get_stats()
        
        return {
            "cache_levels": stats,
            "enabled_strategies": [s.value for s in self.strategies],
            "default_ttl": self.ttl,
        }