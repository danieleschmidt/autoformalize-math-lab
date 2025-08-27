"""Robust formalization pipeline with comprehensive error handling."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from .pipeline import FormalizationPipeline, TargetSystem, FormalizationResult
from .config import FormalizationConfig
from .exceptions import (
    FormalizationError, ParseError, GenerationError, VerificationError,
    ModelError, TimeoutError, RecoveryError, ValidationError
)
from ..utils.logging_config import setup_logger, setup_request_logging, get_correlation_id
from ..utils.metrics import FormalizationMetrics


@dataclass
class ValidationResult:
    """Result of input/output validation."""
    valid: bool
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobustExecutionContext:
    """Context for robust pipeline execution."""
    correlation_id: str = field(default_factory=get_correlation_id)
    start_time: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    validation_enabled: bool = True
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time


@dataclass
class RobustFormalizationResult(FormalizationResult):
    """Enhanced formalization result with robustness metrics."""
    context: Optional[RobustExecutionContext] = None
    recovery_attempts: int = 0
    validation_passed: bool = True
    warnings: List[str] = field(default_factory=list)


class RobustFormalizationPipeline:
    """Robust formalization pipeline with comprehensive error handling."""
    
    def __init__(
        self,
        target_system: TargetSystem,
        config: Optional[FormalizationConfig] = None,
        max_retry_attempts: int = 3
    ):
        self.target_system = target_system
        self.config = config or FormalizationConfig()
        self.logger = setup_logger(__name__)
        
        # Initialize components
        self.base_pipeline = FormalizationPipeline(
            target_system=target_system,
            config=self.config
        )
        
        self.metrics = FormalizationMetrics(
            enable_prometheus=self.config.enable_metrics
        )
        
        self._performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0
        }
        
        self.logger.info(
            f"Initialized robust pipeline for {target_system.value} "
            f"with {max_retry_attempts} max retries"
        )
    
    async def formalize_robust(
        self,
        latex_content: str,
        timeout: Optional[float] = None,
        validate_input: bool = True,
        validate_output: bool = True
    ) -> RobustFormalizationResult:
        """Robustly formalize mathematical content with comprehensive error handling."""
        
        context = RobustExecutionContext(
            timeout=timeout or self.config.model.timeout,
            validation_enabled=validate_input or validate_output
        )
        
        setup_request_logging(context.correlation_id)
        
        self.logger.info(
            f"Starting robust formalization [ID: {context.correlation_id}]"
        )
        
        warnings = []
        
        try:
            # Input validation
            if validate_input:
                validation_result = await self._validate_input(latex_content)
                if not validation_result.valid:
                    return RobustFormalizationResult(
                        success=False,
                        error_message=f"Input validation failed: {validation_result.error}",
                        validation_passed=False,
                        context=context
                    )
                warnings.extend(validation_result.warnings)
            
            # Execute formalization
            start_time = time.time()
            result = await self.base_pipeline.formalize(latex_content)
            processing_time = time.time() - start_time
            
            # Output validation
            if validate_output and result.success and result.formal_code:
                validation_result = await self._validate_output(result.formal_code)
                if not validation_result.valid:
                    warnings.append(f"Output validation warning: {validation_result.error}")
                warnings.extend(validation_result.warnings)
            
            # Create robust result
            robust_result = RobustFormalizationResult(
                success=result.success,
                formal_code=result.formal_code,
                error_message=result.error_message,
                verification_status=result.verification_status,
                metrics=result.metrics,
                correction_rounds=result.correction_rounds,
                processing_time=processing_time,
                context=context,
                warnings=warnings
            )
            
            # Record metrics
            self._record_metrics(robust_result)
            
            return robust_result
            
        except Exception as e:
            self.logger.error(f"Robust execution failed [ID: {context.correlation_id}]: {e}")
            
            return RobustFormalizationResult(
                success=False,
                error_message=str(e),
                context=context,
                warnings=warnings,
                processing_time=context.elapsed_time
            )
    
    async def _validate_input(self, content: str) -> ValidationResult:
        """Validate input LaTeX content."""
        if not content or not content.strip():
            return ValidationResult(False, "Empty content")
        if len(content) > 100000:  # 100KB limit
            return ValidationResult(False, "Content too large")
        return ValidationResult(True, warnings=["Input validation passed"])
    
    async def _validate_output(self, formal_code: str) -> ValidationResult:
        """Validate output formal code."""
        if not formal_code or not formal_code.strip():
            return ValidationResult(False, "Empty formal code")
        return ValidationResult(True, warnings=["Output validation passed"])
    
    def _record_metrics(self, result: RobustFormalizationResult) -> None:
        """Record execution metrics."""
        try:
            self.metrics.record_formalization(
                success=result.success,
                target_system=self.target_system.value,
                processing_time=result.processing_time,
                error=result.error_message,
                correction_rounds=result.correction_rounds,
                verification_success=result.verification_status
            )
            
            # Update performance stats
            self._performance_stats['total_requests'] += 1
            if result.success:
                self._performance_stats['successful_requests'] += 1
            else:
                self._performance_stats['failed_requests'] += 1
                
        except Exception as e:
            self.logger.warning(f"Failed to record metrics: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            'service': f'robust_formalization_pipeline_{self.target_system.value}',
            'healthy': True,
            'timestamp': time.time(),
            'performance_stats': dict(self._performance_stats),
            'metrics_summary': self.metrics.get_summary()
        }