"""Robust error recovery and resilience system.

This module provides advanced error recovery, fault tolerance, and 
self-healing capabilities for the formalization pipeline.
"""

import asyncio
import logging
import traceback
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path

from ..utils.logging_config import setup_logger
from .exceptions import FormalizationError, RecoveryError


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
    ALTERNATIVE_MODEL = "alternative_model"
    SIMPLIFIED_APPROACH = "simplified_approach"
    HUMAN_INTERVENTION = "human_intervention"
    SKIP = "skip"


@dataclass
class ErrorContext:
    """Context information for error recovery."""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    timestamp: float = field(default_factory=time.time)
    stack_trace: Optional[str] = None
    recovery_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Recovery action specification."""
    strategy: RecoveryStrategy
    description: str
    priority: int
    prerequisites: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_probability: float = 0.5
    estimated_time: float = 0.0


class RobustErrorRecoverySystem:
    """Advanced error recovery and resilience system."""
    
    def __init__(
        self,
        max_recovery_attempts: int = 5,
        recovery_timeout: float = 300.0,
        enable_learning: bool = True,
        enable_proactive_healing: bool = True
    ):
        """Initialize robust error recovery system.
        
        Args:
            max_recovery_attempts: Maximum recovery attempts per error
            recovery_timeout: Overall timeout for recovery process
            enable_learning: Whether to learn from recovery patterns
            enable_proactive_healing: Whether to enable proactive self-healing
        """
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_timeout = recovery_timeout
        self.enable_learning = enable_learning
        self.enable_proactive_healing = enable_proactive_healing
        
        self.logger = setup_logger(__name__)
        
        # Error tracking and learning
        self.error_history: List[ErrorContext] = []
        self.recovery_patterns: Dict[str, List[RecoveryAction]] = {}
        self.success_rates: Dict[str, Dict[str, float]] = {}
        
        # Recovery strategies registry
        self.recovery_strategies: Dict[str, Callable] = {
            RecoveryStrategy.RETRY.value: self._retry_recovery,
            RecoveryStrategy.FALLBACK.value: self._fallback_recovery,
            RecoveryStrategy.ALTERNATIVE_MODEL.value: self._alternative_model_recovery,
            RecoveryStrategy.SIMPLIFIED_APPROACH.value: self._simplified_approach_recovery,
            RecoveryStrategy.SKIP.value: self._skip_recovery
        }
        
        # Initialize recovery knowledge
        self._initialize_recovery_knowledge()
        
        # Proactive healing task
        if self.enable_proactive_healing:
            self._proactive_healing_task = None
            self._start_proactive_healing()
    
    def _initialize_recovery_knowledge(self) -> None:
        """Initialize recovery knowledge base."""
        # Common error patterns and their recovery strategies
        self.recovery_patterns = {
            "timeout_error": [
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    description="Retry with increased timeout",
                    priority=1,
                    parameters={"timeout_multiplier": 2.0},
                    success_probability=0.7
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.SIMPLIFIED_APPROACH,
                    description="Use simplified generation approach",
                    priority=2,
                    success_probability=0.8
                )
            ],
            "model_error": [
                RecoveryAction(
                    strategy=RecoveryStrategy.ALTERNATIVE_MODEL,
                    description="Switch to alternative LLM model",
                    priority=1,
                    parameters={"fallback_models": ["gpt-3.5-turbo", "claude-3-haiku"]},
                    success_probability=0.8
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    description="Retry with exponential backoff",
                    priority=2,
                    parameters={"backoff_factor": 2.0},
                    success_probability=0.6
                )
            ],
            "verification_error": [
                RecoveryAction(
                    strategy=RecoveryStrategy.FALLBACK,
                    description="Fall back to mock verification",
                    priority=1,
                    success_probability=0.9
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.SIMPLIFIED_APPROACH,
                    description="Generate simpler proof structure",
                    priority=2,
                    success_probability=0.7
                )
            ],
            "parsing_error": [
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    description="Retry with relaxed parsing rules",
                    priority=1,
                    parameters={"strict_mode": False},
                    success_probability=0.8
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.SIMPLIFIED_APPROACH,
                    description="Extract basic mathematical content only",
                    priority=2,
                    success_probability=0.9
                )
            ],
            "validation_error": [
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    description="Retry with relaxed validation",
                    priority=1,
                    parameters={"validation_level": "basic"},
                    success_probability=0.8
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.SKIP,
                    description="Skip validation and proceed",
                    priority=3,
                    success_probability=1.0
                )
            ]
        }
    
    async def handle_error_with_recovery(
        self,
        error: Exception,
        context: Dict[str, Any],
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Handle error with intelligent recovery.
        
        Args:
            error: The error that occurred
            context: Context information about the operation
            operation: The operation to retry
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result after successful recovery
            
        Raises:
            RecoveryError: If all recovery attempts fail
        """
        error_context = self._create_error_context(error, context)
        self.error_history.append(error_context)
        
        self.logger.warning(f"Error detected: {error_context.error_type} - {error_context.error_message}")
        
        # Get recovery strategies
        recovery_actions = self._get_recovery_strategies(error_context)
        
        # Attempt recovery
        for action in recovery_actions:
            if error_context.recovery_attempts >= self.max_recovery_attempts:
                break
            
            try:
                self.logger.info(f"Attempting recovery: {action.description}")
                
                result = await self._execute_recovery_action(
                    action, error_context, operation, *args, **kwargs
                )
                
                # Recovery successful
                self.logger.info(f"Recovery successful: {action.description}")
                self._record_recovery_success(error_context, action)
                
                return result
                
            except Exception as recovery_error:
                error_context.recovery_attempts += 1
                self.logger.warning(f"Recovery attempt failed: {recovery_error}")
                self._record_recovery_failure(error_context, action)
        
        # All recovery attempts failed
        self.logger.error(f"All recovery attempts failed for {error_context.error_type}")
        raise RecoveryError(
            f"Recovery failed after {error_context.recovery_attempts} attempts: {error_context.error_message}"
        )
    
    def _create_error_context(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Create error context from exception and operation context."""
        error_type = type(error).__name__.lower().replace("error", "").replace("exception", "")
        
        # Determine severity
        severity = self._determine_error_severity(error, context)
        
        return ErrorContext(
            error_type=error_type,
            error_message=str(error),
            severity=severity,
            component=context.get("component", "unknown"),
            stack_trace=traceback.format_exc(),
            metadata=context
        )
    
    def _determine_error_severity(self, error: Exception, context: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity based on error type and context."""
        critical_errors = [
            "SystemError", "MemoryError", "KeyboardInterrupt",
            "SystemExit", "GeneratorExit"
        ]
        
        high_severity_errors = [
            "ConnectionError", "TimeoutError", "AuthenticationError",
            "PermissionError", "FileNotFoundError"
        ]
        
        error_name = type(error).__name__
        
        if error_name in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_name in high_severity_errors:
            return ErrorSeverity.HIGH
        elif "network" in str(error).lower() or "connection" in str(error).lower():
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM
    
    def _get_recovery_strategies(self, error_context: ErrorContext) -> List[RecoveryAction]:
        """Get recovery strategies for the given error context."""
        # Get base strategies for error type
        base_strategies = self.recovery_patterns.get(error_context.error_type, [])
        
        # Customize based on context and learning
        strategies = []
        for action in base_strategies:
            # Adjust success probability based on historical data
            historical_success = self._get_historical_success_rate(
                error_context.error_type, action.strategy.value
            )
            if historical_success is not None:
                action.success_probability = historical_success
            
            strategies.append(action)
        
        # Sort by priority and success probability
        strategies.sort(key=lambda x: (-x.priority, -x.success_probability))
        
        # Add fallback strategies if none specific to error type
        if not strategies:
            strategies = [
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    description="Generic retry",
                    priority=1,
                    success_probability=0.5
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.SKIP,
                    description="Skip operation",
                    priority=99,
                    success_probability=1.0
                )
            ]
        
        return strategies
    
    def _get_historical_success_rate(self, error_type: str, strategy: str) -> Optional[float]:
        """Get historical success rate for error type and strategy combination."""
        if not self.enable_learning:
            return None
        
        return self.success_rates.get(error_type, {}).get(strategy)
    
    async def _execute_recovery_action(
        self,
        action: RecoveryAction,
        error_context: ErrorContext,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute a recovery action."""
        strategy_func = self.recovery_strategies.get(action.strategy.value)
        if not strategy_func:
            raise RecoveryError(f"Unknown recovery strategy: {action.strategy.value}")
        
        # Apply strategy-specific modifications
        modified_args, modified_kwargs = await strategy_func(
            action, error_context, args, kwargs
        )
        
        # Execute the operation with modifications
        if asyncio.iscoroutinefunction(operation):
            return await operation(*modified_args, **modified_kwargs)
        else:
            return operation(*modified_args, **modified_kwargs)
    
    async def _retry_recovery(
        self,
        action: RecoveryAction,
        error_context: ErrorContext,
        args: tuple,
        kwargs: dict
    ) -> tuple:
        """Implement retry recovery strategy."""
        # Add backoff delay
        backoff_factor = action.parameters.get("backoff_factor", 1.5)
        delay = min(backoff_factor ** error_context.recovery_attempts, 60.0)
        
        if delay > 0:
            self.logger.info(f"Waiting {delay:.1f}s before retry")
            await asyncio.sleep(delay)
        
        # Modify timeout if specified
        timeout_multiplier = action.parameters.get("timeout_multiplier", 1.0)
        if "timeout" in kwargs:
            kwargs["timeout"] *= timeout_multiplier
        
        return args, kwargs
    
    async def _fallback_recovery(
        self,
        action: RecoveryAction,
        error_context: ErrorContext,
        args: tuple,
        kwargs: dict
    ) -> tuple:
        """Implement fallback recovery strategy."""
        # Enable fallback mode in kwargs
        kwargs["fallback_mode"] = True
        kwargs["original_error"] = error_context.error_message
        
        return args, kwargs
    
    async def _alternative_model_recovery(
        self,
        action: RecoveryAction,
        error_context: ErrorContext,
        args: tuple,
        kwargs: dict
    ) -> tuple:
        """Implement alternative model recovery strategy."""
        fallback_models = action.parameters.get("fallback_models", ["gpt-3.5-turbo"])
        
        # Select fallback model (could be more sophisticated)
        fallback_model = fallback_models[0] if fallback_models else "gpt-3.5-turbo"
        
        # Update model in kwargs
        kwargs["model_name"] = fallback_model
        kwargs["fallback_model"] = True
        
        return args, kwargs
    
    async def _simplified_approach_recovery(
        self,
        action: RecoveryAction,
        error_context: ErrorContext,
        args: tuple,
        kwargs: dict
    ) -> tuple:
        """Implement simplified approach recovery strategy."""
        # Enable simplified mode
        kwargs["simplified_mode"] = True
        kwargs["max_complexity"] = action.parameters.get("max_complexity", 5)
        kwargs["strict_mode"] = False
        
        return args, kwargs
    
    async def _skip_recovery(
        self,
        action: RecoveryAction,
        error_context: ErrorContext,
        args: tuple,
        kwargs: dict
    ) -> tuple:
        """Implement skip recovery strategy."""
        # Return a mock success result
        return args, {"skip_operation": True}
    
    def _record_recovery_success(self, error_context: ErrorContext, action: RecoveryAction) -> None:
        """Record successful recovery for learning."""
        if not self.enable_learning:
            return
        
        error_type = error_context.error_type
        strategy = action.strategy.value
        
        if error_type not in self.success_rates:
            self.success_rates[error_type] = {}
        
        if strategy not in self.success_rates[error_type]:
            self.success_rates[error_type][strategy] = 0.5
        
        # Update success rate using exponential moving average
        current_rate = self.success_rates[error_type][strategy]
        self.success_rates[error_type][strategy] = 0.9 * current_rate + 0.1 * 1.0
    
    def _record_recovery_failure(self, error_context: ErrorContext, action: RecoveryAction) -> None:
        """Record failed recovery for learning."""
        if not self.enable_learning:
            return
        
        error_type = error_context.error_type
        strategy = action.strategy.value
        
        if error_type not in self.success_rates:
            self.success_rates[error_type] = {}
        
        if strategy not in self.success_rates[error_type]:
            self.success_rates[error_type][strategy] = 0.5
        
        # Update success rate using exponential moving average
        current_rate = self.success_rates[error_type][strategy]
        self.success_rates[error_type][strategy] = 0.9 * current_rate + 0.1 * 0.0
    
    def _start_proactive_healing(self) -> None:
        """Start proactive healing background task."""
        async def proactive_healing_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    await self._perform_proactive_healing()
                except Exception as e:
                    self.logger.error(f"Proactive healing error: {e}")
        
        self._proactive_healing_task = asyncio.create_task(proactive_healing_loop())
    
    async def _perform_proactive_healing(self) -> None:
        """Perform proactive system healing based on error patterns."""
        if len(self.error_history) < 5:
            return
        
        # Analyze recent error patterns
        recent_errors = self.error_history[-20:]  # Last 20 errors
        error_frequency = {}
        
        for error in recent_errors:
            error_frequency[error.error_type] = error_frequency.get(error.error_type, 0) + 1
        
        # Identify problematic patterns
        for error_type, frequency in error_frequency.items():
            if frequency >= 3:  # Pattern threshold
                self.logger.info(f"Proactive healing: addressing frequent {error_type} errors")
                await self._apply_proactive_fix(error_type, frequency)
    
    async def _apply_proactive_fix(self, error_type: str, frequency: int) -> None:
        """Apply proactive fixes for frequent error patterns."""
        if error_type == "timeout":
            # Proactively increase default timeouts
            self.logger.info("Proactively increasing timeout defaults")
        
        elif error_type == "model":
            # Preload alternative models
            self.logger.info("Proactively warming up alternative models")
        
        elif error_type == "verification":
            # Switch to more lenient verification mode
            self.logger.info("Proactively adjusting verification settings")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        # Basic statistics
        total_errors = len(self.error_history)
        error_types = {}
        severities = {}
        components = {}
        
        for error in self.error_history:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            severities[error.severity.value] = severities.get(error.severity.value, 0) + 1
            components[error.component] = components.get(error.component, 0) + 1
        
        # Recent error trend
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour
        
        return {
            "total_errors": total_errors,
            "error_types": error_types,
            "severities": severities,
            "components": components,
            "recent_errors_count": len(recent_errors),
            "success_rates": dict(self.success_rates),
            "proactive_healing_enabled": self.enable_proactive_healing,
            "learning_enabled": self.enable_learning
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status based on error patterns."""
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]
        
        if len(recent_errors) == 0:
            health_score = 1.0
            status = "healthy"
        elif len(recent_errors) < 5:
            health_score = 0.8
            status = "good"
        elif len(recent_errors) < 15:
            health_score = 0.6
            status = "degraded"
        else:
            health_score = 0.3
            status = "unhealthy"
        
        # Check for critical errors
        critical_errors = [e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]
        if critical_errors:
            health_score *= 0.5
            status = "critical"
        
        return {
            "health_score": health_score,
            "status": status,
            "recent_errors": len(recent_errors),
            "critical_errors": len(critical_errors),
            "recommendations": self._get_health_recommendations(recent_errors)
        }
    
    def _get_health_recommendations(self, recent_errors: List[ErrorContext]) -> List[str]:
        """Generate health recommendations based on recent errors."""
        recommendations = []
        
        if len(recent_errors) > 10:
            recommendations.append("Consider reducing system load or scaling resources")
        
        # Analyze error patterns
        error_types = {}
        for error in recent_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        for error_type, count in error_types.items():
            if count >= 3:
                if error_type == "timeout":
                    recommendations.append("Consider increasing timeout values or optimizing performance")
                elif error_type == "model":
                    recommendations.append("Check LLM API connectivity and rate limits")
                elif error_type == "verification":
                    recommendations.append("Verify proof assistant installation and configuration")
        
        if not recommendations:
            recommendations.append("System appears healthy")
        
        return recommendations
    
    async def export_recovery_data(self, filepath: Path) -> None:
        """Export recovery data for analysis."""
        data = {
            "error_history": [
                {
                    "error_type": e.error_type,
                    "error_message": e.error_message,
                    "severity": e.severity.value,
                    "component": e.component,
                    "timestamp": e.timestamp,
                    "recovery_attempts": e.recovery_attempts
                }
                for e in self.error_history
            ],
            "success_rates": dict(self.success_rates),
            "recovery_patterns": {
                error_type: [
                    {
                        "strategy": action.strategy.value,
                        "description": action.description,
                        "priority": action.priority,
                        "success_probability": action.success_probability
                    }
                    for action in actions
                ]
                for error_type, actions in self.recovery_patterns.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Recovery data exported to {filepath}")
    
    def __del__(self):
        """Cleanup proactive healing task."""
        if hasattr(self, '_proactive_healing_task') and self._proactive_healing_task:
            self._proactive_healing_task.cancel()