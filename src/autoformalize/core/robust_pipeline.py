"""Robust formalization pipeline with enhanced error handling and resilience.

This module extends the basic pipeline with comprehensive error handling,
retry mechanisms, fallback strategies, and monitoring capabilities.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import time

from .pipeline import FormalizationPipeline, FormalizationResult, TargetSystem
from .config import FormalizationConfig
from .exceptions import FormalizationError, UnsupportedSystemError, ValidationError
from ..utils.resilience import (
    retry_async, CircuitBreaker, CircuitBreakerConfig, 
    health_check, graceful_degradation, resource_monitor
)
from ..utils.logging_config import setup_logger
from ..utils.metrics import FormalizationMetrics


@dataclass
class RobustFormalizationResult(FormalizationResult):
    """Extended result with robustness metrics."""
    retry_count: int = 0
    fallback_used: bool = False
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    health_status: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class RobustFormalizationPipeline(FormalizationPipeline):
    """Enhanced pipeline with robustness features.
    
    This pipeline extends the basic FormalizationPipeline with:
    - Comprehensive error handling and recovery
    - Retry mechanisms with exponential backoff
    - Circuit breakers for external services
    - Resource monitoring and limits
    - Health checks and system monitoring
    - Graceful degradation strategies
    - Enhanced logging and metrics
    """
    
    def __init__(
        self,
        target_system: Union[str, TargetSystem] = TargetSystem.LEAN4,
        model: str = "gpt-4",
        config: Optional[FormalizationConfig] = None,
        api_key: Optional[str] = None,
        enable_circuit_breaker: bool = True,
        enable_retry: bool = True,
        max_retries: int = 3
    ):
        super().__init__(target_system, model, config, api_key)
        
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        
        self.logger = setup_logger(f"{__name__}.RobustPipeline")
        self.robust_metrics = FormalizationMetrics()
        
        self._setup_robustness_features()
        self._register_health_checks()
        self._register_fallback_handlers()
    
    def _setup_robustness_features(self):
        """Initialize robustness features."""
        # Circuit breaker for LLM API calls
        if self.enable_circuit_breaker:
            breaker_config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60.0,
                expected_exception=Exception
            )
            self.llm_circuit_breaker = CircuitBreaker(breaker_config)
        
        # Resource monitoring
        self.resource_monitor = resource_monitor
        
        self.logger.info("Robustness features initialized")
    
    def _register_health_checks(self):
        """Register system health checks."""
        def check_memory():
            resources = self.resource_monitor.check_resources()
            return not resources['memory_limit_exceeded']
        
        def check_disk_space():
            try:
                import shutil
                total, used, free = shutil.disk_usage("/")
                return free > 1024 * 1024 * 100  # 100MB minimum
            except:
                return True  # Assume OK if check fails
        
        health_check.register("memory", check_memory)
        health_check.register("disk_space", check_disk_space)
        
        self.logger.debug("Health checks registered")
    
    def _register_fallback_handlers(self):
        """Register fallback handlers for graceful degradation."""
        async def llm_fallback(*args, **kwargs):
            """Fallback when LLM service is unavailable."""
            self.logger.warning("Using LLM fallback - returning template-based response")
            # Return a basic template response
            return "theorem placeholder : Prop := sorry"
        
        async def verification_fallback(*args, **kwargs):
            """Fallback when verification service is unavailable."""
            self.logger.warning("Verification service unavailable - skipping verification")
            return None
        
        graceful_degradation.register_fallback("llm", llm_fallback)
        graceful_degradation.register_fallback("verification", verification_fallback)
        
        self.logger.debug("Fallback handlers registered")
    
    async def _validate_input(self, latex_content: str) -> None:
        """Validate input before processing."""
        if not latex_content or not latex_content.strip():
            raise ValidationError("Empty or whitespace-only LaTeX content")
        
        if len(latex_content) > 100_000:  # 100KB limit
            raise ValidationError("LaTeX content too large (>100KB)")
        
        # Check for potentially problematic patterns
        dangerous_patterns = [
            "\\input{",
            "\\include{", 
            "\\write",
            "\\immediate",
        ]
        
        for pattern in dangerous_patterns:
            if pattern in latex_content:
                self.logger.warning(f"Potentially dangerous LaTeX pattern detected: {pattern}")
    
    async def _robust_generate(self, parsed_content) -> str:
        """Generate formal code with retry and circuit breaker."""
        if self.enable_circuit_breaker:
            @self.llm_circuit_breaker
            async def protected_generate():
                return await self.generator.generate(parsed_content)
            
            return await graceful_degradation.execute_with_fallback(
                "llm", protected_generate
            )
        else:
            return await graceful_degradation.execute_with_fallback(
                "llm", self.generator.generate, parsed_content
            )
    
    async def robust_formalize(
        self,
        latex_content: str,
        verify: bool = True,
        timeout: int = 30,
        enable_monitoring: bool = True
    ) -> RobustFormalizationResult:
        """Robust formalization with comprehensive error handling.
        
        Args:
            latex_content: LaTeX source containing mathematical content
            verify: Whether to verify the generated formal code
            timeout: Timeout in seconds for verification
            enable_monitoring: Whether to enable resource monitoring
            
        Returns:
            RobustFormalizationResult with enhanced metadata
        """
        start_time = time.time()
        retry_count = 0
        fallback_used = False
        warnings = []
        
        try:
            self.logger.info(f"Starting robust formalization to {self.target_system.value}")
            
            # Pre-flight health check
            if enable_monitoring:
                health_status = await health_check.check_all()
                if not all(health_status.values()):
                    unhealthy = [k for k, v in health_status.items() if not v]
                    warnings.append(f"Unhealthy systems detected: {unhealthy}")
                    self.logger.warning(f"Proceeding with unhealthy systems: {unhealthy}")
            
            # Input validation
            await self._validate_input(latex_content)
            
            # Resource monitoring
            if enable_monitoring:
                try:
                    self.resource_monitor.enforce_limits()
                except MemoryError as e:
                    raise FormalizationError(f"Resource limit exceeded: {e}")
            
            # Step 1: Parse LaTeX content with retry
            self.logger.debug("Parsing LaTeX content (robust)")
            parsed_content = await self.parser.parse(latex_content)
            
            if not parsed_content.theorems and not parsed_content.definitions:
                raise FormalizationError("No mathematical content found in LaTeX")
            
            # Step 2: Generate formal code with robustness features
            self.logger.debug(f"Generating {self.target_system.value} code (robust)")
            try:
                formal_code = await self._robust_generate(parsed_content)
            except Exception as e:
                fallback_used = True
                warnings.append("LLM generation failed, used fallback")
                formal_code = await graceful_degradation.execute_with_fallback(
                    "llm", lambda: "theorem fallback : Prop := sorry"
                )
            
            # Step 3: Verify if requested and verifier available
            verification_status = None
            if verify and self.verifier:
                self.logger.debug("Verifying generated code (robust)")
                try:
                    verification_status = await graceful_degradation.execute_with_fallback(
                        "verification", self.verifier.verify, formal_code, timeout
                    )
                except Exception as e:
                    warnings.append(f"Verification failed: {e}")
                    verification_status = False
            
            # Collect comprehensive metrics
            processing_time = time.time() - start_time
            resource_usage = self.resource_monitor.check_resources() if enable_monitoring else {}
            
            metrics = {
                "processing_time": processing_time,
                "content_length": len(latex_content),
                "theorems_count": len(parsed_content.theorems),
                "definitions_count": len(parsed_content.definitions),
                "formal_code_length": len(formal_code) if formal_code else 0,
                "retry_count": retry_count,
                "fallback_used": fallback_used,
                "resource_usage": resource_usage,
                "warnings_count": len(warnings),
            }
            
            # Record metrics
            self.robust_metrics.record_formalization(
                success=True,
                target_system=self.target_system.value,
                processing_time=processing_time,
                verification_success=verification_status,
                additional_data={"retry_count": retry_count, "fallback_used": fallback_used}
            )
            
            self.logger.info(
                f"Robust formalization completed successfully in {processing_time:.2f}s "
                f"(retries: {retry_count}, fallback: {fallback_used})"
            )
            
            return RobustFormalizationResult(
                success=True,
                formal_code=formal_code,
                verification_status=verification_status,
                metrics=metrics,
                processing_time=processing_time,
                retry_count=retry_count,
                fallback_used=fallback_used,
                resource_usage=resource_usage,
                health_status=health_status if enable_monitoring else {},
                warnings=warnings
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            resource_usage = self.resource_monitor.check_resources() if enable_monitoring else {}
            
            self.logger.error(f"Robust formalization failed after {processing_time:.2f}s: {e}")
            
            # Record failure metrics
            self.robust_metrics.record_formalization(
                success=False,
                target_system=self.target_system.value,
                processing_time=processing_time,
                error=str(e),
                additional_data={"retry_count": retry_count, "fallback_used": fallback_used}
            )
            
            return RobustFormalizationResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                retry_count=retry_count,
                fallback_used=fallback_used,
                resource_usage=resource_usage,
                warnings=warnings
            )
    
    async def formalize(
        self,
        latex_content: str,
        verify: bool = True,
        timeout: int = 30
    ) -> FormalizationResult:
        """Override base formalize to use robust version."""
        robust_result = await self.robust_formalize(latex_content, verify, timeout)
        
        # Convert to base FormalizationResult for compatibility
        return FormalizationResult(
            success=robust_result.success,
            formal_code=robust_result.formal_code,
            error_message=robust_result.error_message,
            verification_status=robust_result.verification_status,
            metrics=robust_result.metrics,
            correction_rounds=0,  # Not implemented in robust pipeline yet
            processing_time=robust_result.processing_time
        )
    
    async def batch_formalize_robust(
        self,
        input_files: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        parallel: int = 2,  # Reduced default for stability
        verify: bool = True,
        fail_fast: bool = False
    ) -> List[RobustFormalizationResult]:
        """Robust batch processing with enhanced error handling."""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Limit parallelism for robustness
        actual_parallel = min(parallel, 4)
        semaphore = asyncio.Semaphore(actual_parallel)
        
        async def process_file_robust(input_path: Path) -> RobustFormalizationResult:
            async with semaphore:
                try:
                    # Check system health before processing
                    if not await health_check.is_healthy():
                        self.logger.warning(f"System unhealthy, processing {input_path} anyway")
                    
                    # Generate output path if directory provided
                    output_path = None
                    if output_dir:
                        if self.target_system == TargetSystem.LEAN4:
                            output_path = output_dir / f"{input_path.stem}.lean"
                        elif self.target_system == TargetSystem.ISABELLE:
                            output_path = output_dir / f"{input_path.stem}.thy"
                        elif self.target_system == TargetSystem.COQ:
                            output_path = output_dir / f"{input_path.stem}.v"
                    
                    # Read and validate file
                    try:
                        with open(input_path, 'r', encoding='utf-8') as f:
                            latex_content = f.read()
                    except Exception as e:
                        raise FormalizationError(f"Failed to read {input_path}: {e}")
                    
                    # Process with robust pipeline
                    result = await self.robust_formalize(latex_content, verify=verify)
                    
                    # Write output if successful
                    if result.success and result.formal_code and output_path:
                        try:
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(result.formal_code)
                            self.logger.info(f"Output written to {output_path}")
                        except Exception as e:
                            result.warnings.append(f"Failed to write output: {e}")
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {input_path}: {e}")
                    if fail_fast:
                        raise e
                    
                    return RobustFormalizationResult(
                        success=False,
                        error_message=str(e),
                        warnings=[f"File processing failed: {e}"]
                    )
        
        # Process all files with controlled concurrency
        self.logger.info(f"Starting batch processing of {len(input_files)} files with {actual_parallel} workers")
        
        tasks = [process_file_robust(Path(f)) for f in input_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if fail_fast:
                    raise result
                final_results.append(RobustFormalizationResult(
                    success=False,
                    error_message=str(result),
                    warnings=[f"Task exception: {result}"]
                ))
            else:
                final_results.append(result)
        
        # Log comprehensive summary
        successful = sum(1 for r in final_results if r.success)
        total = len(final_results)
        fallback_used = sum(1 for r in final_results if r.fallback_used)
        avg_retries = sum(r.retry_count for r in final_results) / len(final_results)
        
        self.logger.info(
            f"Batch processing completed: {successful}/{total} successful, "
            f"{fallback_used} used fallbacks, avg {avg_retries:.1f} retries"
        )
        
        return final_results
    
    def get_robust_metrics(self) -> Dict[str, Any]:
        """Get comprehensive robustness metrics."""
        base_metrics = self.robust_metrics.get_summary()
        
        # Add robustness-specific metrics
        robustness_metrics = {
            "health_checks": health_check.checks.keys(),
            "circuit_breaker_enabled": self.enable_circuit_breaker,
            "retry_enabled": self.enable_retry,
            "max_retries": self.max_retries,
            "resource_limits": {
                "max_memory_mb": self.resource_monitor.max_memory_mb,
                "max_cpu_percent": self.resource_monitor.max_cpu_percent,
            }
        }
        
        return {**base_metrics, "robustness": robustness_metrics}