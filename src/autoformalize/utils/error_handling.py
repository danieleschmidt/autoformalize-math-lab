"""Advanced error handling and recovery utilities.

This module provides comprehensive error handling, recovery mechanisms,
and diagnostic capabilities for the formalization pipeline.
"""

import sys
import traceback
import functools
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..core.exceptions import (
    AutoformalizeError, FormalizationError, ValidationError,
    VerificationError, GenerationError, TimeoutError, APIError
)
from .logging_config import setup_logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    MANUAL = "manual"


@dataclass
class ErrorContext:
    """Context information for errors."""
    operation: str
    component: str
    input_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    stack_trace: Optional[str] = None
    system_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Recovery action for errors."""
    strategy: RecoveryStrategy
    max_retries: int = 3
    backoff_factor: float = 1.5
    fallback_function: Optional[Callable] = None
    description: str = ""


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    context: ErrorContext
    recovery_action: Optional[RecoveryAction] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    occurrences: int = 1


class ErrorHandler:
    """Comprehensive error handler with recovery capabilities.
    
    This class provides centralized error handling, automatic recovery,
    and detailed error reporting for the formalization pipeline.
    """
    
    def __init__(self, enable_recovery: bool = True):
        """Initialize error handler.
        
        Args:
            enable_recovery: Whether to enable automatic error recovery
        """
        self.enable_recovery = enable_recovery
        self.logger = setup_logger(__name__)
        
        # Error tracking
        self.error_history: List[ErrorRecord] = []
        self.error_counts: Dict[str, int] = {}
        
        # Recovery strategies by error type
        self.recovery_strategies = {
            TimeoutError: RecoveryAction(RecoveryStrategy.RETRY, max_retries=2),
            APIError: RecoveryAction(RecoveryStrategy.RETRY, max_retries=3, backoff_factor=2.0),
            VerificationError: RecoveryAction(RecoveryStrategy.FALLBACK),
            GenerationError: RecoveryAction(RecoveryStrategy.RETRY, max_retries=2),
            ValidationError: RecoveryAction(RecoveryStrategy.ABORT),
            FormalizationError: RecoveryAction(RecoveryStrategy.FALLBACK),
        }
        
        # Error severity mapping
        self.severity_mapping = {
            ValidationError: ErrorSeverity.HIGH,
            APIError: ErrorSeverity.MEDIUM,
            TimeoutError: ErrorSeverity.MEDIUM,
            VerificationError: ErrorSeverity.LOW,
            GenerationError: ErrorSeverity.MEDIUM,
            FormalizationError: ErrorSeverity.HIGH,
            Exception: ErrorSeverity.CRITICAL,
        }
    
    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        raise_on_failure: bool = True
    ) -> Optional[Any]:
        """Handle an error with appropriate recovery strategy.
        
        Args:
            error: Exception that occurred
            context: Context information about the error
            raise_on_failure: Whether to raise exception if recovery fails
            
        Returns:
            Recovery result or None
        """
        try:
            # Determine error severity
            severity = self._get_error_severity(error)
            
            # Create error record
            error_record = self._create_error_record(error, context, severity)
            
            # Log error
            self._log_error(error_record)
            
            # Add to error history
            self.error_history.append(error_record)
            self._update_error_counts(error)
            
            # Attempt recovery if enabled
            if self.enable_recovery:
                recovery_result = self._attempt_recovery(error, error_record)
                if recovery_result is not None:
                    error_record.resolved = True
                    error_record.resolution_time = datetime.now()
                    return recovery_result
            
            # If recovery failed or disabled, raise or return None
            if raise_on_failure:
                raise error
            return None
            
        except Exception as handler_error:
            # Error in error handler - this is critical
            self.logger.critical(f"Error handler failed: {handler_error}")
            if raise_on_failure:
                raise error  # Raise original error
            return None
    
    def create_error_context(
        self,
        operation: str,
        component: str,
        input_data: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> ErrorContext:
        """Create error context for better error tracking.
        
        Args:
            operation: Name of the operation being performed
            component: Component where error occurred
            input_data: Input data that caused the error (sanitized)
            correlation_id: Correlation ID for request tracking
            **kwargs: Additional system information
            
        Returns:
            ErrorContext object
        """
        system_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            **kwargs
        }
        
        return ErrorContext(
            operation=operation,
            component=component,
            input_data=self._sanitize_input_data(input_data),
            correlation_id=correlation_id,
            stack_trace=traceback.format_exc(),
            system_info=system_info
        )
    
    def register_recovery_strategy(
        self,
        error_type: Type[Exception],
        strategy: RecoveryAction
    ) -> None:
        """Register a custom recovery strategy for an error type.
        
        Args:
            error_type: Exception type
            strategy: Recovery action to use
        """
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"Registered recovery strategy for {error_type.__name__}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        total_errors = len(self.error_history)
        if total_errors == 0:
            return {"total_errors": 0}
        
        # Count by severity
        severity_counts = {}
        for record in self.error_history:
            severity = record.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count resolved vs unresolved
        resolved_count = sum(1 for record in self.error_history if record.resolved)
        
        # Most common errors
        error_type_counts = {}
        for record in self.error_history:
            error_type = record.error_type
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        most_common = sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_errors": total_errors,
            "resolved_errors": resolved_count,
            "unresolved_errors": total_errors - resolved_count,
            "resolution_rate": resolved_count / total_errors,
            "severity_distribution": severity_counts,
            "most_common_errors": most_common,
            "error_types": list(error_type_counts.keys())
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent error records.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of recent error records
        """
        recent = self.error_history[-limit:]
        return [self._serialize_error_record(record) for record in recent]
    
    def clear_error_history(self) -> None:
        """Clear error history and statistics."""
        self.error_history.clear()
        self.error_counts.clear()
        self.logger.info("Error history cleared")
    
    def _create_error_record(
        self,
        error: Exception,
        context: ErrorContext,
        severity: ErrorSeverity
    ) -> ErrorRecord:
        """Create an error record from exception and context."""
        error_type = type(error).__name__
        
        # Check if this error type already occurred
        recovery_action = self.recovery_strategies.get(type(error))
        
        return ErrorRecord(
            error_type=error_type,
            error_message=str(error),
            severity=severity,
            context=context,
            recovery_action=recovery_action
        )
    
    def _get_error_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        for error_type, severity in self.severity_mapping.items():
            if isinstance(error, error_type):
                return severity
        
        return ErrorSeverity.MEDIUM  # Default severity
    
    def _log_error(self, error_record: ErrorRecord) -> None:
        """Log error record with appropriate level."""
        context = error_record.context
        
        log_message = (
            f"Error in {context.component}.{context.operation}: "
            f"{error_record.error_message}"
        )
        
        # Add correlation ID if available
        if context.correlation_id:
            log_message = f"[{context.correlation_id}] {log_message}"
        
        # Log with appropriate level based on severity
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Log stack trace for debugging
        if context.stack_trace and error_record.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.debug(f"Stack trace: {context.stack_trace}")
    
    def _attempt_recovery(self, error: Exception, error_record: ErrorRecord) -> Optional[Any]:
        """Attempt to recover from error using registered strategies."""
        recovery_action = error_record.recovery_action
        if not recovery_action:
            return None
        
        self.logger.info(f"Attempting recovery with strategy: {recovery_action.strategy.value}")
        
        try:
            if recovery_action.strategy == RecoveryStrategy.RETRY:
                return self._retry_operation(error, recovery_action)
            elif recovery_action.strategy == RecoveryStrategy.FALLBACK:
                return self._fallback_operation(error, recovery_action)
            elif recovery_action.strategy == RecoveryStrategy.SKIP:
                self.logger.info("Skipping operation due to error")
                return "SKIPPED"
            elif recovery_action.strategy == RecoveryStrategy.ABORT:
                self.logger.error("Aborting operation due to error")
                raise error
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
            return None
        
        return None
    
    def _retry_operation(self, error: Exception, recovery_action: RecoveryAction) -> Optional[Any]:
        """Retry operation with exponential backoff."""
        # This is a placeholder - actual retry logic would need access to the original operation
        self.logger.info(f"Retry recovery attempted for {type(error).__name__}")
        return None
    
    def _fallback_operation(self, error: Exception, recovery_action: RecoveryAction) -> Optional[Any]:
        """Execute fallback operation."""
        if recovery_action.fallback_function:
            try:
                return recovery_action.fallback_function()
            except Exception as fallback_error:
                self.logger.error(f"Fallback operation failed: {fallback_error}")
        
        return None
    
    def _update_error_counts(self, error: Exception) -> None:
        """Update error occurrence counts."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def _sanitize_input_data(self, input_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sanitize input data for logging (remove sensitive information)."""
        if not input_data:
            return None
        
        sanitized = {}
        sensitive_keys = ['api_key', 'password', 'token', 'secret', 'auth']
        
        for key, value in input_data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:997] + "..."
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _serialize_error_record(self, record: ErrorRecord) -> Dict[str, Any]:
        """Serialize error record for JSON output."""
        return {
            "error_type": record.error_type,
            "error_message": record.error_message,
            "severity": record.severity.value,
            "timestamp": record.context.timestamp.isoformat(),
            "operation": record.context.operation,
            "component": record.context.component,
            "correlation_id": record.context.correlation_id,
            "resolved": record.resolved,
            "resolution_time": record.resolution_time.isoformat() if record.resolution_time else None,
            "occurrences": record.occurrences
        }


def with_error_handling(
    operation: str,
    component: str,
    error_handler: Optional[ErrorHandler] = None,
    correlation_id: Optional[str] = None,
    **context_kwargs
):
    """Decorator for automatic error handling.
    
    Args:
        operation: Name of the operation
        component: Component name
        error_handler: ErrorHandler instance (creates default if None)
        correlation_id: Correlation ID for tracking
        **context_kwargs: Additional context information
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            handler = error_handler or ErrorHandler()
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = handler.create_error_context(
                    operation=operation,
                    component=component,
                    input_data={"args": args, "kwargs": kwargs},
                    correlation_id=correlation_id,
                    **context_kwargs
                )
                
                return handler.handle_error(e, context, raise_on_failure=True)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            handler = error_handler or ErrorHandler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = handler.create_error_context(
                    operation=operation,
                    component=component,
                    input_data={"args": args, "kwargs": kwargs},
                    correlation_id=correlation_id,
                    **context_kwargs
                )
                
                return handler.handle_error(e, context, raise_on_failure=True)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global error handler instance
global_error_handler = ErrorHandler()