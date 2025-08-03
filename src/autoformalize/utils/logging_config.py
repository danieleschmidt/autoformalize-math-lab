"""Logging configuration utilities.

This module provides centralized logging setup and configuration
for the autoformalize package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from rich.logging import RichHandler
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    RichHandler = None
    Console = None

try:
    from loguru import logger as loguru_logger
    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False
    loguru_logger = None


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_rich: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup a logger with appropriate handlers and formatting.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        use_rich: Whether to use rich formatting (if available)
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Default format
    if not format_string:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if use_rich and HAS_RICH:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=True,
            markup=True
        )
        console_handler.setFormatter(
            logging.Formatter("%(message)s", datefmt="[%X]")
        )
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler (if requested)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_structured_logging(
    service_name: str = "autoformalize",
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_json: bool = False
) -> logging.Logger:
    """Setup structured logging with JSON output.
    
    Args:
        service_name: Name of the service for log entries
        log_level: Logging level
        log_file: Optional log file path
        enable_json: Whether to use JSON formatting
        
    Returns:
        Configured structured logger
    """
    import json
    import traceback
    
    class StructuredFormatter(logging.Formatter):
        """Custom formatter for structured logging."""
        
        def format(self, record: logging.LogRecord) -> str:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "service": service_name,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = {
                    "type": record.exc_info[0].__name__,
                    "message": str(record.exc_info[1]),
                    "traceback": traceback.format_exception(*record.exc_info)
                }
            
            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in log_entry and not key.startswith('_'):
                    log_entry[key] = value
            
            if enable_json:
                return json.dumps(log_entry)
            else:
                # Human-readable structured format
                return f"[{log_entry['timestamp']}] {log_entry['level']} {log_entry['logger']}: {log_entry['message']}"
    
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    formatter = StructuredFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_correlation_id() -> str:
    """Generate a correlation ID for request tracking."""
    import uuid
    return str(uuid.uuid4())[:8]


class CorrelationFilter(logging.Filter):
    """Filter to add correlation IDs to log records."""
    
    def __init__(self, correlation_id: Optional[str] = None):
        super().__init__()
        self.correlation_id = correlation_id or get_correlation_id()
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = self.correlation_id
        return True


def setup_request_logging(correlation_id: Optional[str] = None) -> str:
    """Setup logging for a specific request/operation.
    
    Args:
        correlation_id: Optional correlation ID (generated if not provided)
        
    Returns:
        The correlation ID used
    """
    if not correlation_id:
        correlation_id = get_correlation_id()
    
    # Add correlation filter to all loggers
    correlation_filter = CorrelationFilter(correlation_id)
    
    # Apply to root logger (affects all child loggers)
    root_logger = logging.getLogger()
    root_logger.addFilter(correlation_filter)
    
    return correlation_id


def configure_performance_logging() -> logging.Logger:
    """Configure logger specifically for performance metrics."""
    perf_logger = logging.getLogger("autoformalize.performance")
    
    if perf_logger.handlers:
        return perf_logger
    
    perf_logger.setLevel(logging.INFO)
    
    # Create performance-specific formatter
    class PerformanceFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            if hasattr(record, 'metric_name') and hasattr(record, 'metric_value'):
                return f"PERF [{record.asctime}] {record.metric_name}={record.metric_value} {record.getMessage()}"
            return super().format(record)
    
    formatter = PerformanceFormatter(
        fmt="PERF [%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    perf_logger.addHandler(handler)
    
    return perf_logger


def log_performance_metric(
    metric_name: str,
    metric_value: float,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> None:
    """Log a performance metric.
    
    Args:
        metric_name: Name of the metric
        metric_value: Value of the metric
        logger: Logger to use (defaults to performance logger)
        **kwargs: Additional context to include
    """
    if not logger:
        logger = configure_performance_logging()
    
    # Create log record with extra attributes
    extra = {
        'metric_name': metric_name,
        'metric_value': metric_value,
        **kwargs
    }
    
    context_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
    message = f"Metric recorded" + (f" ({context_str})" if context_str else "")
    
    logger.info(message, extra=extra)


def silence_noisy_loggers() -> None:
    """Silence commonly noisy third-party loggers."""
    noisy_loggers = [
        "urllib3.connectionpool",
        "requests.packages.urllib3.connectionpool",
        "httpx",
        "httpcore",
        "openai._base_client",
        "anthropic._base_client",
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# Configure default logging when module is imported
silence_noisy_loggers()
