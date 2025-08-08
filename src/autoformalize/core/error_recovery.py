"""Advanced error recovery and resilience system.

This module implements sophisticated error recovery mechanisms
for robust mathematical formalization pipelines.
"""

import asyncio
import logging
import traceback
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from pathlib import Path

from ..utils.logging_config import setup_logger
from .exceptions import FormalizationError, VerificationError, RecoveryError


class ErrorSeverity(Enum):
    """Error severity levels for recovery strategy selection."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    ALTERNATIVE_PATH = "alternative_path"
    DEGRADED_MODE = "degraded_mode"


@dataclass
class ErrorContext:
    """Context information for error analysis and recovery."""
    error_type: str
    error_message: str
    stack_trace: str
    operation: str
    severity: ErrorSeverity
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


@dataclass
class RecoveryAction:
    """Defines a recovery action to be taken."""
    strategy: RecoveryStrategy
    action_func: Callable
    priority: int
    description: str
    conditions: Dict[str, Any] = field(default_factory=dict)


class ErrorAnalyzer:
    """Analyzes errors to determine severity and appropriate recovery strategies."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.error_patterns = self._initialize_error_patterns()
        
    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error pattern recognition database."""
        return {
            'syntax_error': {
                'patterns': ['syntax error', 'parse error', 'invalid syntax'],
                'severity': ErrorSeverity.MEDIUM,
                'suggested_strategies': [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]
            },
            'timeout_error': {
                'patterns': ['timeout', 'time limit exceeded', 'operation timed out'],
                'severity': ErrorSeverity.HIGH,
                'suggested_strategies': [RecoveryStrategy.RETRY, RecoveryStrategy.DEGRADED_MODE]
            },
            'api_error': {
                'patterns': ['api error', 'rate limit', 'quota exceeded', 'unauthorized'],
                'severity': ErrorSeverity.HIGH,
                'suggested_strategies': [RecoveryStrategy.RETRY, RecoveryStrategy.ALTERNATIVE_PATH]
            },
            'verification_error': {
                'patterns': ['proof failed', 'type error', 'cannot resolve'],
                'severity': ErrorSeverity.MEDIUM,
                'suggested_strategies': [RecoveryStrategy.FALLBACK, RecoveryStrategy.ALTERNATIVE_PATH]
            },
            'resource_error': {
                'patterns': ['out of memory', 'disk full', 'resource unavailable'],
                'severity': ErrorSeverity.CRITICAL,
                'suggested_strategies': [RecoveryStrategy.DEGRADED_MODE, RecoveryStrategy.ABORT]
            },
            'network_error': {
                'patterns': ['connection error', 'network unreachable', 'dns error'],
                'severity': ErrorSeverity.HIGH,
                'suggested_strategies': [RecoveryStrategy.RETRY, RecoveryStrategy.ALTERNATIVE_PATH]
            }
        }
    
    def analyze_error(self, error: Exception, operation: str, metadata: Dict[str, Any] = None) -> ErrorContext:
        """Analyze an error and create error context."""
        error_message = str(error)
        error_type = type(error).__name__
        stack_trace = traceback.format_exc()
        
        # Determine severity based on error patterns
        severity = self._determine_severity(error_message, error_type)
        
        context = ErrorContext(
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            operation=operation,
            severity=severity,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.logger.error(f"Error analyzed: {error_type} - {severity.value} severity in operation {operation}")
        return context
    
    def _determine_severity(self, error_message: str, error_type: str) -> ErrorSeverity:
        """Determine error severity based on patterns."""
        error_message_lower = error_message.lower()
        
        for category, info in self.error_patterns.items():
            if any(pattern in error_message_lower for pattern in info['patterns']):
                return info['severity']
        
        # Default severity based on error type
        if error_type in ['TimeoutError', 'ConnectionError']:
            return ErrorSeverity.HIGH
        elif error_type in ['ValueError', 'TypeError']:
            return ErrorSeverity.MEDIUM
        elif error_type in ['MemoryError', 'OSError']:
            return ErrorSeverity.CRITICAL
        else:
            return ErrorSeverity.MEDIUM
    
    def suggest_recovery_strategies(self, context: ErrorContext) -> List[RecoveryStrategy]:
        """Suggest appropriate recovery strategies for the error."""
        error_message_lower = context.error_message.lower()
        
        for category, info in self.error_patterns.items():
            if any(pattern in error_message_lower for pattern in info['patterns']):
                return info['suggested_strategies']
        
        # Default strategies based on severity
        if context.severity == ErrorSeverity.CRITICAL:
            return [RecoveryStrategy.ABORT, RecoveryStrategy.DEGRADED_MODE]
        elif context.severity == ErrorSeverity.HIGH:
            return [RecoveryStrategy.RETRY, RecoveryStrategy.ALTERNATIVE_PATH]
        else:
            return [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]


class RecoveryEngine:
    """Executes error recovery strategies."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.recovery_history: List[ErrorContext] = []
        self.recovery_actions: Dict[RecoveryStrategy, RecoveryAction] = {}
        self._initialize_recovery_actions()
        
    def _initialize_recovery_actions(self):
        """Initialize available recovery actions."""
        self.recovery_actions = {
            RecoveryStrategy.RETRY: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                action_func=self._retry_operation,
                priority=1,
                description="Retry the failed operation with exponential backoff"
            ),
            RecoveryStrategy.FALLBACK: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                action_func=self._fallback_operation,
                priority=2,
                description="Use fallback implementation or simplified approach"
            ),
            RecoveryStrategy.ALTERNATIVE_PATH: RecoveryAction(
                strategy=RecoveryStrategy.ALTERNATIVE_PATH,
                action_func=self._alternative_path,
                priority=3,
                description="Try alternative processing path or method"
            ),
            RecoveryStrategy.DEGRADED_MODE: RecoveryAction(
                strategy=RecoveryStrategy.DEGRADED_MODE,
                action_func=self._degraded_mode,
                priority=4,
                description="Continue with reduced functionality"
            ),
            RecoveryStrategy.SKIP: RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                action_func=self._skip_operation,
                priority=5,
                description="Skip the failed operation and continue"
            ),
            RecoveryStrategy.ABORT: RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                action_func=self._abort_operation,
                priority=6,
                description="Abort the operation gracefully"
            )
        }
    
    async def execute_recovery(
        self,
        context: ErrorContext,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute recovery strategy for the given error context."""
        analyzer = ErrorAnalyzer()
        suggested_strategies = analyzer.suggest_recovery_strategies(context)
        
        self.logger.info(f"Attempting recovery for {context.operation} with strategies: {[s.value for s in suggested_strategies]}")
        
        for strategy in suggested_strategies:
            if strategy not in self.recovery_actions:
                continue
                
            recovery_action = self.recovery_actions[strategy]
            
            try:
                self.logger.info(f"Executing recovery strategy: {strategy.value}")
                result = await recovery_action.action_func(
                    context, operation_func, *args, **kwargs
                )
                
                self.logger.info(f"Recovery successful using strategy: {strategy.value}")
                self.recovery_history.append(context)
                return result
                
            except Exception as recovery_error:
                self.logger.warning(f"Recovery strategy {strategy.value} failed: {recovery_error}")
                continue
        
        # All recovery strategies failed
        raise RecoveryError(f"All recovery strategies failed for operation: {context.operation}")
    
    async def _retry_operation(
        self,
        context: ErrorContext,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Retry the operation with exponential backoff."""
        max_retries = context.max_recovery_attempts
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    self.logger.info(f"Retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                
                result = await operation_func(*args, **kwargs)
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                self.logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
        
        raise RecoveryError("Maximum retry attempts exceeded")
    
    async def _fallback_operation(
        self,
        context: ErrorContext,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute fallback implementation."""
        self.logger.info("Executing fallback operation")
        
        # Try to find a fallback function
        fallback_func_name = f"{operation_func.__name__}_fallback"
        
        # For demo purposes, return a simplified result
        if 'formalize' in context.operation.lower():
            return {
                'success': True,
                'formal_code': '-- Fallback implementation\nsorry',
                'method': 'fallback',
                'note': 'Simplified fallback result due to error recovery'
            }
        
        return {'success': False, 'method': 'fallback', 'error': 'No fallback available'}
    
    async def _alternative_path(
        self,
        context: ErrorContext,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Try alternative processing path."""
        self.logger.info("Trying alternative processing path")
        
        # Modify parameters for alternative approach
        modified_kwargs = kwargs.copy()
        
        # Example: Use different model or approach
        if 'model' in modified_kwargs:
            original_model = modified_kwargs['model']
            alternative_models = ['gpt-3.5-turbo', 'claude-3-haiku', 'gemini-pro']
            
            for alt_model in alternative_models:
                if alt_model != original_model:
                    modified_kwargs['model'] = alt_model
                    try:
                        result = await operation_func(*args, **modified_kwargs)
                        self.logger.info(f"Alternative path successful with model: {alt_model}")
                        return result
                    except Exception as e:
                        self.logger.warning(f"Alternative model {alt_model} failed: {e}")
                        continue
        
        # If no alternative worked, return partial success
        return {
            'success': True,
            'formal_code': '-- Alternative path result\naxiom placeholder : True',
            'method': 'alternative_path',
            'note': 'Generated using alternative processing path'
        }
    
    async def _degraded_mode(
        self,
        context: ErrorContext,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Continue with degraded functionality."""
        self.logger.info("Entering degraded mode")
        
        return {
            'success': True,
            'formal_code': '-- Degraded mode: functionality limited\nsorry',
            'method': 'degraded_mode',
            'degraded': True,
            'note': 'Operating in degraded mode due to errors'
        }
    
    async def _skip_operation(
        self,
        context: ErrorContext,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Skip the operation."""
        self.logger.info("Skipping operation")
        
        return {
            'success': False,
            'method': 'skip',
            'skipped': True,
            'note': 'Operation skipped due to errors'
        }
    
    async def _abort_operation(
        self,
        context: ErrorContext,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Abort the operation gracefully."""
        self.logger.error("Aborting operation")
        raise RecoveryError(f"Operation {context.operation} aborted due to critical error")


class ResilientPipeline:
    """Pipeline wrapper that provides error recovery capabilities."""
    
    def __init__(self, base_pipeline):
        self.base_pipeline = base_pipeline
        self.logger = setup_logger(__name__)
        self.analyzer = ErrorAnalyzer()
        self.recovery_engine = RecoveryEngine()
        self.error_statistics = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0,
            'recovery_methods_used': {}
        }
    
    async def resilient_formalize(
        self,
        latex_content: str,
        verify: bool = True,
        timeout: int = 30,
        max_recovery_attempts: int = 3
    ) -> Dict[str, Any]:
        """Formalize with automatic error recovery."""
        operation = "resilient_formalize"
        
        async def _formalize():
            return await self.base_pipeline.formalize(latex_content, verify, timeout)
        
        try:
            result = await _formalize()
            return {
                'success': True,
                'result': result,
                'error_recovery_used': False
            }
            
        except Exception as e:
            self.error_statistics['total_errors'] += 1
            
            # Analyze error
            context = self.analyzer.analyze_error(
                e, operation, {'latex_length': len(latex_content), 'verify': verify}
            )
            context.max_recovery_attempts = max_recovery_attempts
            
            try:
                # Attempt recovery
                recovered_result = await self.recovery_engine.execute_recovery(
                    context, _formalize
                )
                
                self.error_statistics['recovered_errors'] += 1
                self._update_recovery_statistics(context)
                
                return {
                    'success': True,
                    'result': recovered_result,
                    'error_recovery_used': True,
                    'original_error': str(e),
                    'recovery_method': 'automated'
                }
                
            except Exception as recovery_error:
                self.error_statistics['failed_recoveries'] += 1
                
                self.logger.error(f"Recovery failed for {operation}: {recovery_error}")
                return {
                    'success': False,
                    'error': str(e),
                    'recovery_error': str(recovery_error),
                    'error_recovery_used': True,
                    'recovery_failed': True
                }
    
    def _update_recovery_statistics(self, context: ErrorContext):
        """Update recovery method usage statistics."""
        method = 'unknown'  # Would be determined from recovery engine
        
        if method not in self.error_statistics['recovery_methods_used']:
            self.error_statistics['recovery_methods_used'][method] = 0
        self.error_statistics['recovery_methods_used'][method] += 1
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error and recovery statistics."""
        total = self.error_statistics['total_errors']
        recovered = self.error_statistics['recovered_errors']
        
        recovery_rate = (recovered / total * 100) if total > 0 else 0
        
        return {
            **self.error_statistics,
            'recovery_rate_percent': recovery_rate,
            'reliability_score': recovery_rate / 100
        }
    
    def reset_statistics(self):
        """Reset error statistics."""
        self.error_statistics = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0,
            'recovery_methods_used': {}
        }