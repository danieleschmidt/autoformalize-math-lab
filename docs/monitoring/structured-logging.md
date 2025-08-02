# Structured Logging Configuration

## Overview

Structured logging provides consistent, searchable, and analyzable log data across the autoformalize-math-lab application.

## Configuration

### Python Logging Setup

```python
# src/autoformalize/logging_config.py
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any
import traceback

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_entry["correlation_id"] = record.correlation_id
            
        # Add user context if available
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
            
        # Add formalization context
        if hasattr(record, 'formalization_id'):
            log_entry["formalization_id"] = record.formalization_id
            
        if hasattr(record, 'target_system'):
            log_entry["target_system"] = record.target_system
            
        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
            
        # Add custom fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        return json.dumps(log_entry)

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup structured logging configuration"""
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
    
    return root_logger

# Context managers for enhanced logging
class LoggingContext:
    """Context manager for adding fields to log records"""
    
    def __init__(self, **kwargs):
        self.fields = kwargs
        self.old_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.fields.items():
                setattr(record, key, value)
            return record
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)

# Usage examples
logger = logging.getLogger(__name__)

# Basic structured logging
logger.info("Formalization started", extra={
    'extra_fields': {
        'formalization_id': 'f123',
        'target_system': 'lean4',
        'input_size': 1024
    }
})

# With context manager
with LoggingContext(correlation_id='req-456', user_id='user123'):
    logger.info("Processing formalization request")
    logger.error("Validation failed")
```

### Application Integration

```python
# src/autoformalize/core/pipeline.py
import logging
from .logging_config import LoggingContext
import uuid

logger = logging.getLogger(__name__)

class FormalizationPipeline:
    def __init__(self, target_system: str):
        self.target_system = target_system
    
    def formalize(self, latex_proof: str, correlation_id: str = None):
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        formalization_id = str(uuid.uuid4())
        
        with LoggingContext(
            correlation_id=correlation_id,
            formalization_id=formalization_id,
            target_system=self.target_system
        ):
            logger.info("Starting formalization", extra={
                'extra_fields': {
                    'input_length': len(latex_proof),
                    'stage': 'initialization'
                }
            })
            
            try:
                # Parse LaTeX
                logger.info("Parsing LaTeX input", extra={
                    'extra_fields': {'stage': 'parsing'}
                })
                parsed_proof = self.parse_latex(latex_proof)
                
                # Generate formal proof
                logger.info("Generating formal proof", extra={
                    'extra_fields': {'stage': 'generation'}
                })
                formal_proof = self.generate_formal_proof(parsed_proof)
                
                # Verify proof
                logger.info("Verifying formal proof", extra={
                    'extra_fields': {'stage': 'verification'}
                })
                verification_result = self.verify_proof(formal_proof)
                
                if verification_result.success:
                    logger.info("Formalization completed successfully", extra={
                        'extra_fields': {
                            'stage': 'completion',
                            'output_length': len(formal_proof),
                            'verification_time': verification_result.duration
                        }
                    })
                else:
                    logger.warning("Formalization verification failed", extra={
                        'extra_fields': {
                            'stage': 'verification_failed',
                            'error_count': len(verification_result.errors)
                        }
                    })
                
                return formal_proof
                
            except Exception as e:
                logger.error("Formalization failed", exc_info=True, extra={
                    'extra_fields': {
                        'stage': 'error',
                        'error_type': type(e).__name__
                    }
                })
                raise
```

## Log Levels and Usage

### Level Guidelines

- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational information
- **WARNING**: Something unexpected happened but system continues
- **ERROR**: Error occurred but system may continue
- **CRITICAL**: Serious error occurred, system may not continue

### Domain-Specific Logging

```python
# Formalization-specific log categories
FORMALIZATION_LOGGER = logging.getLogger('autoformalize.formalization')
VERIFICATION_LOGGER = logging.getLogger('autoformalize.verification')
PARSING_LOGGER = logging.getLogger('autoformalize.parsing')
LLM_LOGGER = logging.getLogger('autoformalize.llm')

# Usage examples
FORMALIZATION_LOGGER.info("Starting new formalization batch", extra={
    'extra_fields': {
        'batch_size': 10,
        'target_systems': ['lean4', 'isabelle'],
        'priority': 'high'
    }
})

VERIFICATION_LOGGER.error("Proof verification timeout", extra={
    'extra_fields': {
        'proof_id': 'p456',
        'timeout_seconds': 30,
        'proof_length': 150
    }
})

LLM_LOGGER.info("LLM API request completed", extra={
    'extra_fields': {
        'model': 'gpt-4',
        'tokens_used': 1500,
        'response_time_ms': 2300,
        'cost_usd': 0.045
    }
})
```

## Log Collection and Analysis

### ELK Stack Configuration

```yaml
# logging/logstash.conf
input {
  file {
    path => "/var/log/autoformalize/*.log"
    codec => "json"
    type => "autoformalize"
  }
}

filter {
  if [type] == "autoformalize" {
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    if [exception] {
      mutate {
        add_tag => ["error", "exception"]
      }
    }
    
    if [target_system] {
      mutate {
        add_field => { "system_family" => "%{target_system}" }
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "autoformalize-logs-%{+YYYY.MM.dd}"
  }
}
```

### Prometheus Log Metrics

```python
# Log metrics for monitoring
from prometheus_client import Counter, Histogram

log_messages_total = Counter('log_messages_total',
                           'Total log messages by level',
                           ['level', 'logger'])

formalization_duration = Histogram('formalization_duration_seconds',
                                 'Time spent on formalization',
                                 ['target_system', 'success'])

verification_attempts = Counter('verification_attempts_total',
                              'Total verification attempts',
                              ['target_system', 'result'])

# Integration with logging
class MetricsHandler(logging.Handler):
    def emit(self, record):
        log_messages_total.labels(
            level=record.levelname,
            logger=record.name
        ).inc()
        
        if hasattr(record, 'formalization_duration'):
            formalization_duration.labels(
                target_system=getattr(record, 'target_system', 'unknown'),
                success=str(getattr(record, 'success', False))
            ).observe(record.formalization_duration)
```

## Best Practices

1. **Consistent Structure**: Use standardized field names across all logs
2. **Correlation IDs**: Include correlation IDs to trace requests across services
3. **Performance**: Avoid logging sensitive information
4. **Sampling**: Use log sampling for high-volume operations
5. **Retention**: Configure appropriate log retention policies
6. **Monitoring**: Set up alerts on error rates and patterns
7. **Documentation**: Document custom fields and their meanings