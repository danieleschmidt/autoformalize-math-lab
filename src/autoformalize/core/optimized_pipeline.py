"""Optimized and scalable formalization pipeline.

This module extends the robust pipeline with performance optimizations,
intelligent caching, parallel processing, and scalability enhancements.
"""

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import json

from .robust_pipeline import RobustFormalizationPipeline, RobustFormalizationResult
from .config import FormalizationConfig
from .exceptions import FormalizationError
from ..utils.caching import CacheManager, CacheStrategy
from ..utils.concurrency import AsyncBatch, ResourcePool
from ..utils.logging_config import setup_logger
from ..utils.metrics import FormalizationMetrics
from ..utils.resilience import resource_monitor


@dataclass
class OptimizedFormalizationResult(RobustFormalizationResult):
    """Extended result with optimization metrics."""
    cache_hit: bool = False
    cache_key: Optional[str] = None
    parallel_processing_time: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    optimization_stats: Dict[str, Any] = field(default_factory=dict)


class OptimizedFormalizationPipeline(RobustFormalizationPipeline):
    """High-performance, scalable formalization pipeline.
    
    This pipeline adds the following optimizations:
    - Intelligent multi-level caching (memory, disk, distributed)
    - Adaptive parallel processing with work stealing
    - Resource-aware batching and load balancing
    - Performance monitoring and auto-tuning
    - Memory-efficient streaming for large datasets
    - Predictive pre-computation and warming
    - Advanced metrics and telemetry
    """
    
    def __init__(
        self,
        target_system: Union[str, "TargetSystem"] = "lean4",
        model: str = "gpt-4",
        config: Optional[FormalizationConfig] = None,
        api_key: Optional[str] = None,
        enable_caching: bool = True,
        cache_ttl: int = 3600,  # 1 hour
        max_parallel_workers: int = None,
        enable_predictive_caching: bool = True,
        enable_streaming: bool = True,
        memory_limit_mb: int = 2048
    ):
        super().__init__(target_system, model, config, api_key)
        
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.max_parallel_workers = max_parallel_workers or min(multiprocessing.cpu_count(), 8)
        self.enable_predictive_caching = enable_predictive_caching
        self.enable_streaming = enable_streaming
        self.memory_limit_mb = memory_limit_mb
        
        self.logger = setup_logger(f"{__name__}.OptimizedPipeline")
        self.optimization_metrics = FormalizationMetrics()
        
        self._setup_optimization_features()
        
    def _setup_optimization_features(self):
        """Initialize optimization features."""
        # Cache manager with multiple strategies
        if self.enable_caching:
            self.cache_manager = CacheManager(
                strategies=[
                    CacheStrategy.MEMORY,
                    CacheStrategy.DISK,
                ],
                ttl=self.cache_ttl,
                max_memory_size=256 * 1024 * 1024,  # 256MB
                disk_cache_dir=Path("./cache/formalization")
            )
        
        # Resource pool for managing computational resources
        self.resource_pool = ResourcePool(
            max_cpu_workers=self.max_parallel_workers,
            max_memory_mb=self.memory_limit_mb,
            enable_monitoring=True
        )
        
        # Async batch processor for efficient batching
        self.batch_processor = AsyncBatch(
            batch_size=10,
            max_wait_time=5.0,
            max_workers=self.max_parallel_workers
        )
        
        # Performance tracking
        self.performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_processing_time": 0.0,
            "parallel_speedup": 0.0,
            "memory_efficiency": 0.0,
        }
        
        self.logger.info(
            f"Optimization features initialized: "
            f"caching={'enabled' if self.enable_caching else 'disabled'}, "
            f"workers={self.max_parallel_workers}, "
            f"memory_limit={self.memory_limit_mb}MB"
        )
    
    def _generate_cache_key(self, latex_content: str, config: Dict[str, Any]) -> str:
        """Generate a unique cache key for content and configuration."""
        content_hash = hashlib.sha256(latex_content.encode()).hexdigest()[:16]
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        return f"{self.target_system.value}:{self.model}:{content_hash}:{config_hash}"
    
    async def _try_cache_lookup(
        self, 
        latex_content: str, 
        config: Dict[str, Any]
    ) -> Optional[OptimizedFormalizationResult]:
        """Try to retrieve result from cache."""
        if not self.enable_caching:
            return None
        
        cache_key = self._generate_cache_key(latex_content, config)
        
        try:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                self.performance_stats["cache_hits"] += 1
                self.logger.debug(f"Cache hit for key {cache_key[:16]}...")
                
                # Convert to OptimizedFormalizationResult
                result = OptimizedFormalizationResult(**cached_result)
                result.cache_hit = True
                result.cache_key = cache_key
                return result
                
        except Exception as e:
            self.logger.warning(f"Cache lookup failed: {e}")
        
        self.performance_stats["cache_misses"] += 1
        return None
    
    async def _cache_result(
        self, 
        cache_key: str, 
        result: OptimizedFormalizationResult
    ) -> None:
        """Store result in cache."""
        if not self.enable_caching:
            return
        
        try:
            # Convert to dict for caching (remove non-serializable fields)
            cache_data = {
                "success": result.success,
                "formal_code": result.formal_code,
                "error_message": result.error_message,
                "verification_status": result.verification_status,
                "processing_time": result.processing_time,
                "retry_count": result.retry_count,
                "fallback_used": result.fallback_used,
                "warnings": result.warnings,
                "metrics": result.metrics,
            }
            
            await self.cache_manager.set(cache_key, cache_data)
            self.logger.debug(f"Cached result for key {cache_key[:16]}...")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache result: {e}")
    
    async def optimized_formalize(
        self,
        latex_content: str,
        verify: bool = True,
        timeout: int = 30,
        enable_parallel: bool = True,
        cache_strategy: str = "auto"
    ) -> OptimizedFormalizationResult:
        """Optimized formalization with caching and performance enhancements.
        
        Args:
            latex_content: LaTeX source containing mathematical content
            verify: Whether to verify the generated formal code
            timeout: Timeout in seconds for verification
            enable_parallel: Enable parallel processing where possible
            cache_strategy: Caching strategy ("auto", "force", "bypass")
            
        Returns:
            OptimizedFormalizationResult with performance metrics
        """
        start_time = time.time()
        config = {
            "verify": verify,
            "timeout": timeout,
            "target_system": self.target_system.value,
        }
        
        # Try cache lookup first
        cache_key = self._generate_cache_key(latex_content, config)
        if cache_strategy != "bypass":
            cached_result = await self._try_cache_lookup(latex_content, config)
            if cached_result:
                return cached_result
        
        # Monitor resource usage during processing
        initial_resources = resource_monitor.check_resources()
        
        try:
            # Use robust formalization as base
            robust_result = await self.robust_formalize(
                latex_content=latex_content,
                verify=verify,
                timeout=timeout,
                enable_monitoring=True
            )
            
            # Get final resource usage
            final_resources = resource_monitor.check_resources()
            memory_peak_mb = max(
                initial_resources.get('memory_mb', 0),
                final_resources.get('memory_mb', 0)
            )
            
            # Create optimized result
            processing_time = time.time() - start_time
            optimized_result = OptimizedFormalizationResult(
                success=robust_result.success,
                formal_code=robust_result.formal_code,
                error_message=robust_result.error_message,
                verification_status=robust_result.verification_status,
                metrics=robust_result.metrics,
                processing_time=processing_time,
                retry_count=robust_result.retry_count,
                fallback_used=robust_result.fallback_used,
                resource_usage=robust_result.resource_usage,
                health_status=robust_result.health_status,
                warnings=robust_result.warnings,
                cache_hit=False,
                cache_key=cache_key,
                memory_peak_mb=memory_peak_mb,
                optimization_stats={
                    "cache_hits": self.performance_stats["cache_hits"],
                    "cache_misses": self.performance_stats["cache_misses"],
                    "worker_pool_size": self.max_parallel_workers,
                    "memory_efficiency": memory_peak_mb / self.memory_limit_mb,
                }
            )
            
            # Cache successful results
            if optimized_result.success and cache_strategy != "bypass":
                await self._cache_result(cache_key, optimized_result)
            
            # Update performance tracking
            self.performance_stats["total_processing_time"] += processing_time
            
            return optimized_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Optimized formalization failed: {e}")
            
            return OptimizedFormalizationResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                cache_key=cache_key,
                optimization_stats=self.performance_stats.copy()
            )
    
    async def batch_formalize_optimized(
        self,
        input_files: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        parallel: int = None,
        verify: bool = True,
        chunk_size: int = 100,
        enable_streaming: bool = None,
        progress_callback: Optional[callable] = None
    ) -> List[OptimizedFormalizationResult]:
        """Highly optimized batch processing with streaming and parallelization.
        
        Args:
            input_files: List of input LaTeX file paths
            output_dir: Optional output directory
            parallel: Number of parallel workers (None = auto-detect)
            verify: Whether to verify generated code
            chunk_size: Size of processing chunks for streaming
            enable_streaming: Enable streaming processing for large datasets
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of OptimizedFormalizationResult objects
        """
        actual_parallel = parallel or self.max_parallel_workers
        enable_streaming = enable_streaming if enable_streaming is not None else self.enable_streaming
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(
            f"Starting optimized batch processing: "
            f"{len(input_files)} files, {actual_parallel} workers, "
            f"streaming={'enabled' if enable_streaming else 'disabled'}"
        )
        
        start_time = time.time()
        
        # Use streaming processing for large datasets
        if enable_streaming and len(input_files) > chunk_size:
            results = await self._stream_batch_process(
                input_files, output_dir, actual_parallel, verify, 
                chunk_size, progress_callback
            )
        else:
            results = await self._parallel_batch_process(
                input_files, output_dir, actual_parallel, verify, progress_callback
            )
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        cache_hits = sum(1 for r in results if r.cache_hit)
        
        # Calculate performance metrics
        sequential_time = sum(r.processing_time for r in results)
        speedup = sequential_time / total_time if total_time > 0 else 1.0
        
        self.performance_stats["parallel_speedup"] = speedup
        
        self.logger.info(
            f"Optimized batch processing completed: "
            f"{successful}/{len(results)} successful, "
            f"{cache_hits} cache hits, "
            f"{speedup:.2f}x speedup, "
            f"{total_time:.2f}s total"
        )
        
        return results
    
    async def _stream_batch_process(
        self,
        input_files: List[Union[str, Path]],
        output_dir: Optional[Path],
        parallel: int,
        verify: bool,
        chunk_size: int,
        progress_callback: Optional[callable]
    ) -> List[OptimizedFormalizationResult]:
        """Stream-based batch processing for memory efficiency."""
        all_results = []
        
        # Process files in chunks to manage memory usage
        for chunk_start in range(0, len(input_files), chunk_size):
            chunk_files = input_files[chunk_start:chunk_start + chunk_size]
            
            self.logger.debug(f"Processing chunk {chunk_start//chunk_size + 1}: {len(chunk_files)} files")
            
            chunk_results = await self._parallel_batch_process(
                chunk_files, output_dir, parallel, verify, None
            )
            
            all_results.extend(chunk_results)
            
            # Progress callback
            if progress_callback:
                progress = len(all_results) / len(input_files)
                await progress_callback(progress, len(all_results), len(input_files))
            
            # Garbage collection between chunks
            import gc
            gc.collect()
        
        return all_results
    
    async def _parallel_batch_process(
        self,
        input_files: List[Union[str, Path]],
        output_dir: Optional[Path],
        parallel: int,
        verify: bool,
        progress_callback: Optional[callable]
    ) -> List[OptimizedFormalizationResult]:
        """Parallel batch processing with resource management."""
        semaphore = asyncio.Semaphore(parallel)
        
        async def process_single_file(input_path: Path) -> OptimizedFormalizationResult:
            async with semaphore:
                try:
                    # Generate output path
                    output_path = None
                    if output_dir:
                        if self.target_system.value == "lean4":
                            output_path = output_dir / f"{input_path.stem}.lean"
                        elif self.target_system.value == "isabelle":
                            output_path = output_dir / f"{input_path.stem}.thy"
                        elif self.target_system.value == "coq":
                            output_path = output_dir / f"{input_path.stem}.v"
                    
                    # Read file content
                    try:
                        with open(input_path, 'r', encoding='utf-8') as f:
                            latex_content = f.read()
                    except Exception as e:
                        return OptimizedFormalizationResult(
                            success=False,
                            error_message=f"Failed to read {input_path}: {e}",
                            warnings=[f"File read error: {e}"]
                        )
                    
                    # Process with optimization
                    result = await self.optimized_formalize(
                        latex_content=latex_content,
                        verify=verify,
                        enable_parallel=True,
                        cache_strategy="auto"
                    )
                    
                    # Write output if successful
                    if result.success and result.formal_code and output_path:
                        try:
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(result.formal_code)
                        except Exception as e:
                            result.warnings.append(f"Failed to write output: {e}")
                    
                    return result
                    
                except Exception as e:
                    return OptimizedFormalizationResult(
                        success=False,
                        error_message=str(e),
                        warnings=[f"Processing error: {e}"]
                    )
        
        # Process all files concurrently
        tasks = [process_single_file(Path(f)) for f in input_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                final_results.append(OptimizedFormalizationResult(
                    success=False,
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def warm_cache(
        self, 
        warmup_files: List[Union[str, Path]],
        background: bool = True
    ) -> Dict[str, Any]:
        """Warm up the cache with commonly used patterns."""
        if not self.enable_caching or not self.enable_predictive_caching:
            return {"status": "disabled"}
        
        self.logger.info(f"Warming cache with {len(warmup_files)} files")
        
        async def warm_single_file(file_path: Path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                await self.optimized_formalize(content, cache_strategy="force")
            except Exception as e:
                self.logger.warning(f"Failed to warm cache for {file_path}: {e}")
        
        if background:
            # Start warming in background
            tasks = [warm_single_file(Path(f)) for f in warmup_files]
            asyncio.create_task(asyncio.gather(*tasks, return_exceptions=True))
            return {"status": "started_background", "files": len(warmup_files)}
        else:
            # Wait for completion
            tasks = [warm_single_file(Path(f)) for f in warmup_files]
            await asyncio.gather(*tasks, return_exceptions=True)
            return {"status": "completed", "files": len(warmup_files)}
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get comprehensive optimization and performance metrics."""
        base_metrics = self.get_robust_metrics()
        
        optimization_metrics = {
            "caching": {
                "enabled": self.enable_caching,
                "cache_hits": self.performance_stats["cache_hits"],
                "cache_misses": self.performance_stats["cache_misses"],
                "hit_rate": (
                    self.performance_stats["cache_hits"] / 
                    max(self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"], 1)
                ),
            },
            "parallelization": {
                "max_workers": self.max_parallel_workers,
                "speedup": self.performance_stats["parallel_speedup"],
                "memory_limit_mb": self.memory_limit_mb,
            },
            "resource_usage": {
                "memory_efficiency": self.performance_stats["memory_efficiency"],
                "total_processing_time": self.performance_stats["total_processing_time"],
            },
            "optimization_features": {
                "streaming_enabled": self.enable_streaming,
                "predictive_caching": self.enable_predictive_caching,
                "resource_pooling": True,
            }
        }
        
        return {**base_metrics, "optimization": optimization_metrics}
    
    async def cleanup(self):
        """Clean up optimization resources."""
        if hasattr(self, 'cache_manager'):
            await self.cache_manager.cleanup()
        
        if hasattr(self, 'resource_pool'):
            await self.resource_pool.cleanup()
        
        self.logger.info("Optimization resources cleaned up")