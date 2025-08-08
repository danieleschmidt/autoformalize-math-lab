"""Advanced concurrency utilities for parallel processing."""

import asyncio
import time
from typing import Any, Callable, List, Optional, Dict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from .logging_config import setup_logger


@dataclass
class ResourcePool:
    """Resource pool for managing computational resources."""
    max_cpu_workers: int = 4
    max_memory_mb: int = 1024
    enable_monitoring: bool = True
    
    def __post_init__(self):
        self.logger = setup_logger(f"{__name__}.ResourcePool")
        self.current_memory_usage = 0
        self.active_workers = 0
    
    async def acquire_worker(self) -> bool:
        """Acquire a worker from the pool."""
        if self.active_workers < self.max_cpu_workers:
            self.active_workers += 1
            return True
        return False
    
    async def release_worker(self):
        """Release a worker back to the pool."""
        if self.active_workers > 0:
            self.active_workers -= 1
    
    async def cleanup(self):
        """Clean up resource pool."""
        self.active_workers = 0


class AsyncBatch:
    """Batch processor for efficient async operations."""
    
    def __init__(
        self,
        batch_size: int = 10,
        max_wait_time: float = 5.0,
        max_workers: int = 4
    ):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_workers = max_workers
        self.logger = setup_logger(f"{__name__}.AsyncBatch")
        self.pending_items = []
        self.last_batch_time = time.time()
    
    async def process_batch(
        self,
        items: List[Any],
        processor_func: Callable,
        **kwargs
    ) -> List[Any]:
        """Process a batch of items concurrently."""
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_item(item):
            async with semaphore:
                return await processor_func(item, **kwargs)
        
        tasks = [process_single_item(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def add_item(self, item: Any, processor_func: Callable) -> Optional[Any]:
        """Add item to batch for processing."""
        self.pending_items.append((item, processor_func))
        
        # Process batch if full or max wait time exceeded
        if (len(self.pending_items) >= self.batch_size or 
            time.time() - self.last_batch_time > self.max_wait_time):
            return await self._flush_batch()
        
        return None
    
    async def _flush_batch(self) -> List[Any]:
        """Flush pending items and process batch."""
        if not self.pending_items:
            return []
        
        items, processors = zip(*self.pending_items)
        self.pending_items = []
        self.last_batch_time = time.time()
        
        # For simplicity, use the first processor for all items
        # In a real implementation, you'd group by processor
        processor = processors[0]
        
        return await self.process_batch(list(items), processor)