"""Distributed formalization pipeline for high-throughput processing.

This module implements a distributed architecture for mathematical
formalization that can scale across multiple nodes and handle
high-volume workloads efficiently.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import uuid
from pathlib import Path
import concurrent.futures

try:
    import redis
    from celery import Celery
    HAS_CELERY = True
    HAS_REDIS = True
except ImportError:
    HAS_CELERY = False
    HAS_REDIS = False

from ..core.pipeline import FormalizationPipeline, FormalizationResult
from ..utils.logging_config import setup_logger
from ..utils.metrics import FormalizationMetrics


class TaskStatus(Enum):
    """Status of distributed tasks."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerType(Enum):
    """Types of distributed workers."""
    GENERAL = "general"
    PARSING = "parsing"
    GENERATION = "generation"
    VERIFICATION = "verification"
    OPTIMIZATION = "optimization"


@dataclass
class DistributedTask:
    """Represents a task in the distributed pipeline."""
    task_id: str
    task_type: str
    input_data: Dict[str, Any]
    priority: int = 1
    timeout: float = 300.0
    retries: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    worker_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'input_data': self.input_data,
            'priority': self.priority,
            'timeout': self.timeout,
            'retries': self.retries,
            'max_retries': self.max_retries,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'status': self.status.value,
            'worker_id': self.worker_id,
            'result': self.result,
            'error': self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedTask':
        task = cls(
            task_id=data['task_id'],
            task_type=data['task_type'],
            input_data=data['input_data'],
            priority=data.get('priority', 1),
            timeout=data.get('timeout', 300.0),
            retries=data.get('retries', 0),
            max_retries=data.get('max_retries', 3),
            created_at=data.get('created_at', time.time()),
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            worker_id=data.get('worker_id'),
            result=data.get('result'),
            error=data.get('error')
        )
        task.status = TaskStatus(data.get('status', 'pending'))
        return task


class TaskQueue:
    """High-performance distributed task queue."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.logger = setup_logger(__name__)
        self.redis_url = redis_url
        self.redis_client = None
        self.queue_name = "formalization_tasks"
        self.result_queue = "formalization_results"
        
        if HAS_REDIS:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                self.logger.info("Connected to Redis for task queue")
            except Exception as e:
                self.logger.warning(f"Redis not available: {e}")
                self.redis_client = None
        
        # Fallback in-memory queue
        self._memory_queue: List[DistributedTask] = []
        self._memory_results: Dict[str, DistributedTask] = {}
    
    async def enqueue_task(self, task: DistributedTask) -> bool:
        """Add a task to the queue."""
        try:
            if self.redis_client:
                # Use Redis for distributed queue
                task_data = json.dumps(task.to_dict())
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.lpush, self.queue_name, task_data
                )
                self.logger.debug(f"Task {task.task_id} enqueued to Redis")
                return True
            else:
                # Use in-memory queue as fallback
                self._memory_queue.append(task)
                self._memory_queue.sort(key=lambda t: (-t.priority, t.created_at))
                self.logger.debug(f"Task {task.task_id} enqueued to memory")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to enqueue task {task.task_id}: {e}")
            return False
    
    async def dequeue_task(self, worker_id: str, timeout: float = 10.0) -> Optional[DistributedTask]:
        """Get the next task from the queue."""
        try:
            if self.redis_client:
                # Use Redis blocking pop
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.brpop, self.queue_name, int(timeout)
                )
                if result:
                    _, task_data = result
                    task_dict = json.loads(task_data.decode('utf-8'))
                    task = DistributedTask.from_dict(task_dict)
                    task.worker_id = worker_id
                    task.status = TaskStatus.PROCESSING
                    task.started_at = time.time()
                    return task
            else:
                # Use in-memory queue
                if self._memory_queue:
                    task = self._memory_queue.pop(0)
                    task.worker_id = worker_id
                    task.status = TaskStatus.PROCESSING
                    task.started_at = time.time()
                    return task
                    
        except Exception as e:
            self.logger.error(f"Failed to dequeue task: {e}")
            
        return None
    
    async def complete_task(self, task: DistributedTask) -> bool:
        """Mark a task as completed and store result."""
        try:
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            if self.redis_client:
                # Store result in Redis
                result_data = json.dumps(task.to_dict())
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.setex, 
                    f"result:{task.task_id}", 3600, result_data
                )
            else:
                # Store in memory
                self._memory_results[task.task_id] = task
                
            self.logger.debug(f"Task {task.task_id} completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to complete task {task.task_id}: {e}")
            return False
    
    async def fail_task(self, task: DistributedTask, error: str) -> bool:
        """Mark a task as failed."""
        try:
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = time.time()
            
            # Check if task should be retried
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                task.worker_id = None
                task.started_at = None
                task.completed_at = None
                task.error = None
                
                # Re-enqueue for retry
                await self.enqueue_task(task)
                self.logger.info(f"Task {task.task_id} requeued for retry {task.retries}/{task.max_retries}")
                return True
            else:
                # Store failed result
                if self.redis_client:
                    result_data = json.dumps(task.to_dict())
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.setex,
                        f"result:{task.task_id}", 3600, result_data
                    )
                else:
                    self._memory_results[task.task_id] = task
                    
                self.logger.error(f"Task {task.task_id} failed permanently: {error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to handle task failure {task.task_id}: {e}")
            return False
    
    async def get_task_result(self, task_id: str) -> Optional[DistributedTask]:
        """Get the result of a completed task."""
        try:
            if self.redis_client:
                result_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, f"result:{task_id}"
                )
                if result_data:
                    task_dict = json.loads(result_data.decode('utf-8'))
                    return DistributedTask.from_dict(task_dict)
            else:
                return self._memory_results.get(task_id)
                
        except Exception as e:
            self.logger.error(f"Failed to get task result {task_id}: {e}")
            
        return None
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            if self.redis_client:
                queue_length = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.llen, self.queue_name
                )
                
                # Count results by status
                result_keys = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.keys, "result:*"
                )
                
                return {
                    'pending_tasks': queue_length,
                    'completed_tasks': len(result_keys),
                    'backend': 'redis'
                }
            else:
                pending = len(self._memory_queue)
                completed = len(self._memory_results)
                
                return {
                    'pending_tasks': pending,
                    'completed_tasks': completed,
                    'backend': 'memory'
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get queue stats: {e}")
            return {'error': str(e)}


class DistributedWorker:
    """Worker node for distributed formalization processing."""
    
    def __init__(
        self,
        worker_id: str = None,
        worker_type: WorkerType = WorkerType.GENERAL,
        queue: TaskQueue = None,
        max_concurrent_tasks: int = 4
    ):
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.worker_type = worker_type
        self.queue = queue or TaskQueue()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.logger = setup_logger(__name__)
        
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None
        
        # Initialize pipeline for processing
        self.pipeline = FormalizationPipeline()
        
    async def start(self):
        """Start the worker."""
        if self.running:
            self.logger.warning("Worker already running")
            return
            
        self.running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        self.logger.info(f"Worker {self.worker_id} started (type: {self.worker_type.value})")
    
    async def stop(self):
        """Stop the worker gracefully."""
        if not self.running:
            return
            
        self.running = False
        
        # Cancel worker task
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active tasks to complete
        if self.active_tasks:
            self.logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete")
            await asyncio.sleep(5.0)  # Grace period
            
        self.logger.info(f"Worker {self.worker_id} stopped")
    
    async def _worker_loop(self):
        """Main worker processing loop."""
        while self.running:
            try:
                # Check if we can accept more tasks
                if len(self.active_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(1.0)
                    continue
                
                # Get next task from queue
                task = await self.queue.dequeue_task(self.worker_id, timeout=5.0)
                
                if task:
                    # Process task asynchronously
                    self.active_tasks[task.task_id] = task
                    asyncio.create_task(self._process_task(task))
                else:
                    # No tasks available, wait briefly
                    await asyncio.sleep(1.0)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_task(self, task: DistributedTask):
        """Process a single task."""
        try:
            self.logger.info(f"Processing task {task.task_id} (type: {task.task_type})")
            
            # Route task to appropriate handler
            if task.task_type == "formalize":
                result = await self._handle_formalization_task(task)
            elif task.task_type == "parse":
                result = await self._handle_parsing_task(task)
            elif task.task_type == "generate":
                result = await self._handle_generation_task(task)
            elif task.task_type == "verify":
                result = await self._handle_verification_task(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Store result and complete task
            task.result = result
            await self.queue.complete_task(task)
            
            self.logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {e}")
            await self.queue.fail_task(task, str(e))
            
        finally:
            # Remove from active tasks
            self.active_tasks.pop(task.task_id, None)
    
    async def _handle_formalization_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Handle full formalization task."""
        input_data = task.input_data
        latex_content = input_data.get('latex_content', '')
        target_system = input_data.get('target_system', 'lean4')
        verify = input_data.get('verify', True)
        
        # Create pipeline instance for this task
        pipeline = FormalizationPipeline(target_system=target_system)
        
        # Process formalization
        result = await pipeline.formalize(latex_content, verify=verify)
        
        return {
            'success': result.success,
            'formal_code': result.formal_code,
            'verification_status': result.verification_status,
            'processing_time': result.processing_time,
            'worker_id': self.worker_id
        }
    
    async def _handle_parsing_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Handle LaTeX parsing task."""
        input_data = task.input_data
        latex_content = input_data.get('latex_content', '')
        
        # Simulate parsing (would use actual parser)
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            'success': True,
            'parsed_content': {
                'theorems': ['Sample theorem'],
                'definitions': ['Sample definition'],
                'proofs': ['Sample proof']
            },
            'worker_id': self.worker_id
        }
    
    async def _handle_generation_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Handle code generation task."""
        input_data = task.input_data
        parsed_content = input_data.get('parsed_content', {})
        target_system = input_data.get('target_system', 'lean4')
        
        # Simulate code generation
        await asyncio.sleep(0.5)  # Simulate processing time
        
        return {
            'success': True,
            'formal_code': f'-- Generated for {target_system}\ntheorem sample : True := trivial',
            'worker_id': self.worker_id
        }
    
    async def _handle_verification_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Handle proof verification task."""
        input_data = task.input_data
        formal_code = input_data.get('formal_code', '')
        target_system = input_data.get('target_system', 'lean4')
        
        # Simulate verification
        await asyncio.sleep(0.3)  # Simulate processing time
        
        return {
            'success': True,
            'verification_passed': True,
            'worker_id': self.worker_id
        }


class DistributedFormalizationPipeline:
    """High-performance distributed formalization pipeline."""
    
    def __init__(
        self,
        num_workers: int = 4,
        redis_url: str = "redis://localhost:6379/0",
        auto_scale: bool = True
    ):
        self.logger = setup_logger(__name__)
        self.num_workers = num_workers
        self.auto_scale = auto_scale
        
        # Initialize components
        self.queue = TaskQueue(redis_url)
        self.workers: List[DistributedWorker] = []
        self.metrics = FormalizationMetrics()
        
        # Performance tracking
        self.active_requests: Dict[str, DistributedTask] = {}
        self.throughput_history: List[float] = []
        
    async def start(self):
        """Start the distributed pipeline."""
        self.logger.info(f"Starting distributed pipeline with {self.num_workers} workers")
        
        # Start workers
        for i in range(self.num_workers):
            worker = DistributedWorker(
                worker_id=f"worker-{i}",
                queue=self.queue,
                max_concurrent_tasks=4
            )
            self.workers.append(worker)
            await worker.start()
        
        self.logger.info("Distributed pipeline started successfully")
    
    async def stop(self):
        """Stop the distributed pipeline."""
        self.logger.info("Stopping distributed pipeline")
        
        # Stop all workers
        stop_tasks = [worker.stop() for worker in self.workers]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.workers.clear()
        self.logger.info("Distributed pipeline stopped")
    
    async def formalize_distributed(
        self,
        latex_content: str,
        target_system: str = "lean4",
        verify: bool = True,
        priority: int = 1
    ) -> str:
        """Submit formalization task to distributed pipeline."""
        task_id = f"formalize-{uuid.uuid4().hex[:8]}"
        
        task = DistributedTask(
            task_id=task_id,
            task_type="formalize",
            input_data={
                'latex_content': latex_content,
                'target_system': target_system,
                'verify': verify
            },
            priority=priority
        )
        
        # Enqueue task
        success = await self.queue.enqueue_task(task)
        if not success:
            raise RuntimeError(f"Failed to enqueue task {task_id}")
        
        self.active_requests[task_id] = task
        self.logger.info(f"Formalization task {task_id} submitted")
        
        return task_id
    
    async def get_result(self, task_id: str, timeout: float = 300.0) -> Dict[str, Any]:
        """Get the result of a formalization task."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result_task = await self.queue.get_task_result(task_id)
            
            if result_task:
                if result_task.status == TaskStatus.COMPLETED:
                    self.active_requests.pop(task_id, None)
                    return result_task.result
                elif result_task.status == TaskStatus.FAILED:
                    self.active_requests.pop(task_id, None)
                    raise RuntimeError(f"Task {task_id} failed: {result_task.error}")
            
            await asyncio.sleep(1.0)
        
        raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
    
    async def formalize_and_wait(
        self,
        latex_content: str,
        target_system: str = "lean4",
        verify: bool = True,
        timeout: float = 300.0
    ) -> Dict[str, Any]:
        """Submit formalization task and wait for result."""
        task_id = await self.formalize_distributed(
            latex_content, target_system, verify
        )
        
        return await self.get_result(task_id, timeout)
    
    async def batch_formalize(
        self,
        latex_contents: List[str],
        target_system: str = "lean4",
        verify: bool = True,
        timeout: float = 600.0
    ) -> List[Dict[str, Any]]:
        """Process multiple formalization tasks in parallel."""
        self.logger.info(f"Starting batch formalization of {len(latex_contents)} items")
        
        # Submit all tasks
        task_ids = []
        for latex_content in latex_contents:
            task_id = await self.formalize_distributed(
                latex_content, target_system, verify
            )
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            try:
                result = await self.get_result(task_id, timeout)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'task_id': task_id
                })
        
        return results
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        queue_stats = await self.queue.get_queue_stats()
        
        # Worker statistics
        worker_stats = {
            'total_workers': len(self.workers),
            'active_workers': sum(1 for w in self.workers if w.running),
            'total_active_tasks': sum(len(w.active_tasks) for w in self.workers)
        }
        
        # Performance metrics
        performance_stats = {
            'active_requests': len(self.active_requests),
            'throughput_history': self.throughput_history[-100:],  # Last 100 measurements
        }
        
        return {
            'timestamp': time.time(),
            'queue': queue_stats,
            'workers': worker_stats,
            'performance': performance_stats
        }
    
    async def scale_workers(self, target_workers: int):
        """Dynamically scale the number of workers."""
        current_workers = len(self.workers)
        
        if target_workers > current_workers:
            # Scale up
            for i in range(current_workers, target_workers):
                worker = DistributedWorker(
                    worker_id=f"worker-{i}",
                    queue=self.queue,
                    max_concurrent_tasks=4
                )
                self.workers.append(worker)
                await worker.start()
                
            self.logger.info(f"Scaled up from {current_workers} to {target_workers} workers")
            
        elif target_workers < current_workers:
            # Scale down
            workers_to_stop = self.workers[target_workers:]
            self.workers = self.workers[:target_workers]
            
            stop_tasks = [worker.stop() for worker in workers_to_stop]
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            
            self.logger.info(f"Scaled down from {current_workers} to {target_workers} workers")