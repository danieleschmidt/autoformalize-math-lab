"""Concurrent processing utilities for the formalization pipeline.

This module provides advanced concurrency features including parallel processing,
resource pooling, and adaptive load balancing for optimal performance.
"""

import asyncio
import threading
import multiprocessing
from typing import Dict, List, Optional, Any, Callable, Union, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
import time

from .logging_config import setup_logger
from .metrics import MetricEvent


class ProcessingMode(Enum):
    """Processing modes for different workload types."""
    SEQUENTIAL = "sequential"
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    ASYNC_CONCURRENT = "async_concurrent"
    ADAPTIVE = "adaptive"


@dataclass
class WorkItem:
    """Work item for processing queues."""
    id: str
    task: Union[Callable, Coroutine]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority > other.priority  # Higher priority first


@dataclass
class ProcessingResult:
    """Result of processing a work item."""
    work_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    worker_id: Optional[str] = None


class ResourcePool:
    """Resource pool for managing limited resources like API clients.
    
    This class implements a pool of resources (e.g., HTTP clients, database
    connections) that can be shared across concurrent operations.
    """
    
    def __init__(
        self,
        resource_factory: Callable,
        max_resources: int = 10,
        min_resources: int = 2,
        resource_lifetime: int = 3600  # 1 hour
    ):
        """Initialize resource pool.
        
        Args:
            resource_factory: Function to create new resources
            max_resources: Maximum number of resources in pool
            min_resources: Minimum number of resources to maintain
            resource_lifetime: Resource lifetime in seconds
        """
        self.resource_factory = resource_factory
        self.max_resources = max_resources
        self.min_resources = min_resources
        self.resource_lifetime = resource_lifetime
        self.logger = setup_logger(__name__)
        
        # Resource management
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=max_resources)
        self._resource_info: Dict[int, datetime] = {}
        self._active_resources: int = 0
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "created": 0,
            "destroyed": 0,
            "borrowed": 0,
            "returned": 0,
            "active": 0,
            "pool_size": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the resource pool with minimum resources."""
        async with self._lock:
            for _ in range(self.min_resources):
                await self._create_resource()
    
    async def acquire(self, timeout: float = 30.0) -> Any:
        """Acquire a resource from the pool.
        
        Args:
            timeout: Maximum wait time for resource
            
        Returns:
            Resource object
        """
        try:
            # Try to get existing resource
            resource = await asyncio.wait_for(
                self._pool.get(),
                timeout=timeout
            )
            
            # Check if resource is still valid
            resource_id = id(resource)
            if resource_id in self._resource_info:
                created_at = self._resource_info[resource_id]
                if datetime.now() - created_at > timedelta(seconds=self.resource_lifetime):
                    # Resource expired, create new one
                    await self._destroy_resource(resource)
                    resource = await self._create_resource_direct()
            
            self.stats["borrowed"] += 1
            return resource
            
        except asyncio.TimeoutError:
            # Create new resource if pool is empty and under limit
            async with self._lock:
                if self._active_resources < self.max_resources:
                    resource = await self._create_resource_direct()
                    self.stats["borrowed"] += 1
                    return resource
            
            raise RuntimeError("Resource pool exhausted and timeout reached")
    
    async def release(self, resource: Any) -> None:
        """Release a resource back to the pool.
        
        Args:
            resource: Resource to release
        """
        try:
            # Check if resource is still valid
            resource_id = id(resource)
            if resource_id in self._resource_info:
                created_at = self._resource_info[resource_id]
                if datetime.now() - created_at <= timedelta(seconds=self.resource_lifetime):
                    # Resource still valid, return to pool
                    await self._pool.put(resource)
                    self.stats["returned"] += 1
                    return
            
            # Resource invalid, destroy it
            await self._destroy_resource(resource)
            
            # Create replacement if below minimum
            async with self._lock:
                if self._active_resources < self.min_resources:
                    await self._create_resource()
                    
        except Exception as e:
            self.logger.error(f"Error releasing resource: {e}")
            await self._destroy_resource(resource)
    
    async def _create_resource(self) -> Any:
        """Create a new resource and add to pool."""
        resource = await self._create_resource_direct()
        await self._pool.put(resource)
        return resource
    
    async def _create_resource_direct(self) -> Any:
        """Create a new resource directly."""
        try:
            if asyncio.iscoroutinefunction(self.resource_factory):
                resource = await self.resource_factory()
            else:
                resource = self.resource_factory()
            
            self._resource_info[id(resource)] = datetime.now()
            self._active_resources += 1
            self.stats["created"] += 1
            self.stats["active"] = self._active_resources
            self.stats["pool_size"] = self._pool.qsize()
            
            return resource
            
        except Exception as e:
            self.logger.error(f"Failed to create resource: {e}")
            raise
    
    async def _destroy_resource(self, resource: Any) -> None:
        """Destroy a resource."""
        try:
            resource_id = id(resource)
            if resource_id in self._resource_info:
                del self._resource_info[resource_id]
            
            # Call cleanup method if available
            if hasattr(resource, 'close'):
                if asyncio.iscoroutinefunction(resource.close):
                    await resource.close()
                else:
                    resource.close()
            
            self._active_resources -= 1
            self.stats["destroyed"] += 1
            self.stats["active"] = self._active_resources
            self.stats["pool_size"] = self._pool.qsize()
            
        except Exception as e:
            self.logger.error(f"Error destroying resource: {e}")
    
    async def cleanup(self) -> None:
        """Clean up all resources in the pool."""
        while not self._pool.empty():
            try:
                resource = self._pool.get_nowait()
                await self._destroy_resource(resource)
            except asyncio.QueueEmpty:
                break
        
        self._resource_info.clear()
        self._active_resources = 0


class ConcurrentProcessor:
    """High-performance concurrent processor with adaptive load balancing.
    
    This class provides various concurrency modes and automatically
    adapts to workload characteristics for optimal performance.
    """
    
    def __init__(
        self,
        max_workers: int = None,
        processing_mode: ProcessingMode = ProcessingMode.ADAPTIVE,
        enable_metrics: bool = True
    ):
        """Initialize concurrent processor.
        
        Args:
            max_workers: Maximum number of concurrent workers
            processing_mode: Processing mode to use
            enable_metrics: Whether to collect performance metrics
        """
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.processing_mode = processing_mode
        self.enable_metrics = enable_metrics
        self.logger = setup_logger(__name__)
        
        # Executors
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None
        
        # Work queues
        self.work_queue: asyncio.Queue = asyncio.Queue()
        self.result_queue: asyncio.Queue = asyncio.Queue()
        
        # Worker management
        self.workers: List[asyncio.Task] = []
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        self._shutdown_event = asyncio.Event()
        
        # Performance metrics
        self.metrics = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "avg_processing_time": 0.0,
            "queue_size": 0,
            "active_workers": 0,
            "mode_switches": 0
        }
        
        # Adaptive processing state
        self.performance_history: List[float] = []
        self.last_mode_switch = datetime.now()
    
    async def start(self) -> None:
        """Start the concurrent processor."""
        self.logger.info(f"Starting concurrent processor with {self.max_workers} workers")
        
        # Initialize executors based on mode
        if self.processing_mode in [ProcessingMode.THREAD_POOL, ProcessingMode.ADAPTIVE]:
            self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        if self.processing_mode in [ProcessingMode.PROCESS_POOL, ProcessingMode.ADAPTIVE]:
            self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker_id = f"worker_{i}"
            worker = asyncio.create_task(self._worker_loop(worker_id))
            self.workers.append(worker)
            self.worker_stats[worker_id] = {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "last_active": datetime.now()
            }
    
    async def stop(self) -> None:
        """Stop the concurrent processor."""
        self.logger.info("Stopping concurrent processor")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Cleanup executors
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
        
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
    
    async def submit_task(
        self,
        task: Union[Callable, Coroutine],
        *args,
        priority: int = 0,
        **kwargs
    ) -> str:
        """Submit a task for processing.
        
        Args:
            task: Task function or coroutine to execute
            *args: Task arguments
            priority: Task priority (higher = more important)
            **kwargs: Task keyword arguments
            
        Returns:
            Task ID for tracking
        """
        work_item = WorkItem(
            id=f"task_{int(time.time() * 1000000)}_{len(self.worker_stats)}",
            task=task,
            args=args,
            kwargs=kwargs,
            priority=priority
        )
        
        await self.work_queue.put(work_item)
        self.metrics["queue_size"] = self.work_queue.qsize()
        
        return work_item.id
    
    async def get_result(self, timeout: Optional[float] = None) -> ProcessingResult:
        """Get a processing result.
        
        Args:
            timeout: Maximum wait time for result
            
        Returns:
            ProcessingResult object
        """
        if timeout:
            return await asyncio.wait_for(self.result_queue.get(), timeout)
        else:
            return await self.result_queue.get()
    
    async def process_batch(
        self,
        tasks: List[Union[Callable, Coroutine]],
        args_list: List[tuple] = None,
        kwargs_list: List[dict] = None,
        timeout: Optional[float] = None
    ) -> List[ProcessingResult]:
        """Process a batch of tasks concurrently.
        
        Args:
            tasks: List of tasks to process
            args_list: List of arguments for each task
            kwargs_list: List of keyword arguments for each task
            timeout: Maximum time to wait for all results
            
        Returns:
            List of ProcessingResult objects
        """
        if not tasks:
            return []
        
        # Submit all tasks
        task_ids = []
        for i, task in enumerate(tasks):
            args = args_list[i] if args_list else ()
            kwargs = kwargs_list[i] if kwargs_list else {}
            
            task_id = await self.submit_task(task, *args, **kwargs)
            task_ids.append(task_id)
        
        # Collect results
        results = []
        start_time = time.time()
        
        for _ in task_ids:
            try:
                remaining_timeout = None
                if timeout:
                    elapsed = time.time() - start_time
                    remaining_timeout = max(0, timeout - elapsed)
                    if remaining_timeout <= 0:
                        break
                
                result = await self.get_result(remaining_timeout)
                results.append(result)
                
            except asyncio.TimeoutError:
                self.logger.warning("Batch processing timeout reached")
                break
        
        return results
    
    async def _worker_loop(self, worker_id: str) -> None:
        """Main worker loop."""
        self.logger.debug(f"Worker {worker_id} started")
        
        while not self._shutdown_event.is_set():
            try:
                # Get work item with timeout
                work_item = await asyncio.wait_for(
                    self.work_queue.get(),
                    timeout=1.0
                )
                
                # Process the work item
                result = await self._process_work_item(work_item, worker_id)
                
                # Put result in result queue
                await self.result_queue.put(result)
                
                # Update statistics
                self._update_worker_stats(worker_id, result)
                
                # Adaptive mode switching
                if self.processing_mode == ProcessingMode.ADAPTIVE:
                    await self._check_adaptive_mode_switch()
                
            except asyncio.TimeoutError:
                # No work available, continue
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
        
        self.logger.debug(f"Worker {worker_id} stopped")
    
    async def _process_work_item(self, work_item: WorkItem, worker_id: str) -> ProcessingResult:
        """Process a single work item."""
        start_time = time.time()
        
        try:
            # Execute the task based on current mode
            if asyncio.iscoroutine(work_item.task):
                result = await work_item.task
            elif asyncio.iscoroutinefunction(work_item.task):
                result = await work_item.task(*work_item.args, **work_item.kwargs)
            else:
                # Sync function - execute based on processing mode
                result = await self._execute_sync_task(work_item)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                work_id=work_item.id,
                success=True,
                result=result,
                processing_time=processing_time,
                worker_id=worker_id
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Task {work_item.id} failed: {e}")
            
            return ProcessingResult(
                work_id=work_item.id,
                success=False,
                error=str(e),
                processing_time=processing_time,
                worker_id=worker_id
            )
    
    async def _execute_sync_task(self, work_item: WorkItem) -> Any:
        """Execute synchronous task based on processing mode."""
        if self.processing_mode == ProcessingMode.THREAD_POOL and self.thread_executor:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.thread_executor,
                lambda: work_item.task(*work_item.args, **work_item.kwargs)
            )
        
        elif self.processing_mode == ProcessingMode.PROCESS_POOL and self.process_executor:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.process_executor,
                lambda: work_item.task(*work_item.args, **work_item.kwargs)
            )
        
        else:
            # Sequential execution
            return work_item.task(*work_item.args, **work_item.kwargs)
    
    def _update_worker_stats(self, worker_id: str, result: ProcessingResult) -> None:
        """Update worker statistics."""
        stats = self.worker_stats[worker_id]
        
        if result.success:
            stats["tasks_completed"] += 1
            self.metrics["tasks_processed"] += 1
        else:
            stats["tasks_failed"] += 1
            self.metrics["tasks_failed"] += 1
        
        stats["total_time"] += result.processing_time
        stats["avg_time"] = stats["total_time"] / (stats["tasks_completed"] + stats["tasks_failed"])
        stats["last_active"] = datetime.now()
        
        # Update global metrics
        total_tasks = self.metrics["tasks_processed"] + self.metrics["tasks_failed"]
        if total_tasks > 0:
            total_time = sum(s["total_time"] for s in self.worker_stats.values())
            self.metrics["avg_processing_time"] = total_time / total_tasks
        
        self.metrics["queue_size"] = self.work_queue.qsize()
        self.metrics["active_workers"] = len([
            s for s in self.worker_stats.values()
            if datetime.now() - s["last_active"] < timedelta(seconds=10)
        ])
    
    async def _check_adaptive_mode_switch(self) -> None:
        """Check if processing mode should be switched for better performance."""
        # Only switch if enough time has passed since last switch
        if datetime.now() - self.last_mode_switch < timedelta(seconds=30):
            return
        
        # Collect recent performance data
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]  # Keep last 50
        
        current_avg = self.metrics["avg_processing_time"]
        self.performance_history.append(current_avg)
        
        if len(self.performance_history) < 10:
            return  # Not enough data
        
        # Simple heuristic: if performance is degrading, try different mode
        recent_avg = sum(self.performance_history[-5:]) / 5
        older_avg = sum(self.performance_history[-10:-5]) / 5
        
        if recent_avg > older_avg * 1.2:  # 20% performance degradation
            await self._switch_processing_mode()
    
    async def _switch_processing_mode(self) -> None:
        """Switch to a different processing mode."""
        current_mode = self.processing_mode
        
        # Simple mode rotation for adaptive processing
        if current_mode == ProcessingMode.ASYNC_CONCURRENT:
            new_mode = ProcessingMode.THREAD_POOL
        elif current_mode == ProcessingMode.THREAD_POOL:
            new_mode = ProcessingMode.PROCESS_POOL
        else:
            new_mode = ProcessingMode.ASYNC_CONCURRENT
        
        self.processing_mode = new_mode
        self.last_mode_switch = datetime.now()
        self.metrics["mode_switches"] += 1
        
        self.logger.info(f"Switched processing mode from {current_mode.value} to {new_mode.value}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            **self.metrics,
            "processing_mode": self.processing_mode.value,
            "worker_count": len(self.workers),
            "worker_stats": self.worker_stats,
            "performance_trend": {
                "recent_avg": sum(self.performance_history[-5:]) / 5 if len(self.performance_history) >= 5 else 0,
                "overall_avg": sum(self.performance_history) / len(self.performance_history) if self.performance_history else 0,
                "sample_count": len(self.performance_history)
            }
        }


# Global concurrent processor instance
global_processor = ConcurrentProcessor()