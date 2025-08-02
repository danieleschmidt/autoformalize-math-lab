# Prometheus Metrics Configuration

## Overview

This document outlines the Prometheus metrics collection strategy for monitoring the autoformalize-math-lab application's performance, reliability, and business metrics.

## Core Metrics

### Application Metrics

```python
# src/autoformalize/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from functools import wraps

# Business Metrics
formalizations_total = Counter(
    'formalizations_total',
    'Total number of formalization attempts',
    ['target_system', 'status', 'domain']
)

formalization_success_rate = Gauge(
    'formalization_success_rate',
    'Success rate of formalizations',
    ['target_system', 'domain']
)

formalization_duration_seconds = Histogram(
    'formalization_duration_seconds',
    'Time spent on formalization',
    ['target_system', 'stage'],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, float('inf')]
)

correction_rounds = Histogram(
    'correction_rounds',
    'Number of correction rounds per formalization',
    ['target_system'],
    buckets=[0, 1, 2, 3, 4, 5, 10, float('inf')]
)

# LLM API Metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['provider', 'model', 'status']
)

llm_tokens_used_total = Counter(
    'llm_tokens_used_total',
    'Total tokens consumed by LLM',
    ['provider', 'model', 'type']  # type: input, output
)

llm_cost_usd_total = Counter(
    'llm_cost_usd_total',
    'Total cost in USD for LLM usage',
    ['provider', 'model']
)

llm_response_duration_seconds = Histogram(
    'llm_response_duration_seconds',
    'LLM API response time',
    ['provider', 'model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float('inf')]
)

# Proof Assistant Metrics
proof_verification_duration_seconds = Histogram(
    'proof_verification_duration_seconds',
    'Time spent verifying proofs',
    ['system'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf')]
)

proof_verification_attempts = Counter(
    'proof_verification_attempts_total',
    'Total proof verification attempts',
    ['system', 'result']  # result: success, timeout, error
)

mathlib_usage = Counter(
    'mathlib_usage_total',
    'Mathlib theorem usage count',
    ['theorem_category', 'target_system']
)

# System Metrics
active_formalizations = Gauge(
    'active_formalizations',
    'Currently active formalization processes'
)

error_rate = Gauge(
    'error_rate',
    'Current error rate',
    ['component']
)

# Parser Metrics
latex_parsing_duration_seconds = Histogram(
    'latex_parsing_duration_seconds',
    'Time spent parsing LaTeX input',
    buckets=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, float('inf')]
)

parsing_errors_total = Counter(
    'parsing_errors_total',
    'Total parsing errors',
    ['error_type']
)

# Decorator for timing functions
def time_function(metric_name: str, labels: dict = None):
    """Decorator to time function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                status = 'success'
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                metric = globals().get(metric_name)
                if metric and labels:
                    metric.labels(**labels, status=status).observe(duration)
                elif metric:
                    metric.observe(duration)
        return wrapper
    return decorator

# Context manager for active operations
class ActiveOperationTracker:
    def __init__(self, gauge_metric):
        self.gauge = gauge_metric
    
    def __enter__(self):
        self.gauge.inc()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.gauge.dec()

# Usage examples
@time_function('formalization_duration_seconds', 
               {'target_system': 'lean4', 'stage': 'generation'})
def generate_formal_proof(latex_input: str):
    # Implementation here
    pass

# Track active operations
with ActiveOperationTracker(active_formalizations):
    # Perform formalization
    pass
```

### Integration with Application

```python
# src/autoformalize/core/pipeline.py
import logging
from .metrics import *

class FormalizationPipeline:
    def __init__(self, target_system: str):
        self.target_system = target_system
        self.logger = logging.getLogger(__name__)
    
    def formalize(self, latex_proof: str, domain: str = 'general'):
        start_time = time.time()
        
        with ActiveOperationTracker(active_formalizations):
            try:
                # Parse LaTeX
                with time_function('latex_parsing_duration_seconds'):
                    parsed = self.parse_latex(latex_proof)
                
                # Generate formal proof
                formal_proof = self.generate_formal_proof(parsed)
                
                # Verify proof
                verification_start = time.time()
                verification_result = self.verify_proof(formal_proof)
                verification_duration = time.time() - verification_start
                
                # Record metrics
                proof_verification_duration_seconds.labels(
                    system=self.target_system
                ).observe(verification_duration)
                
                if verification_result.success:
                    status = 'success'
                    proof_verification_attempts.labels(
                        system=self.target_system,
                        result='success'
                    ).inc()
                else:
                    status = 'failed'
                    proof_verification_attempts.labels(
                        system=self.target_system,
                        result='error'
                    ).inc()
                
                # Record formalization attempt
                formalizations_total.labels(
                    target_system=self.target_system,
                    status=status,
                    domain=domain
                ).inc()
                
                # Record duration
                total_duration = time.time() - start_time
                formalization_duration_seconds.labels(
                    target_system=self.target_system,
                    stage='total'
                ).observe(total_duration)
                
                # Update success rate
                self.update_success_rate(domain)
                
                return formal_proof
                
            except Exception as e:
                formalizations_total.labels(
                    target_system=self.target_system,
                    status='error',
                    domain=domain
                ).inc()
                raise
    
    def update_success_rate(self, domain: str):
        # Calculate rolling success rate
        # Implementation depends on your metrics storage
        pass
```

## Metrics Exposition

### HTTP Endpoint

```python
# src/autoformalize/metrics_server.py
from prometheus_client import start_http_server, generate_latest
from flask import Flask, Response

def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics HTTP server"""
    start_http_server(port)

# Flask integration
app = Flask(__name__)

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### Docker Configuration

```dockerfile
# Expose metrics port
EXPOSE 8000

# Health check that includes metrics availability
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/metrics || exit 1
```

## Prometheus Configuration

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "autoformalize_rules.yml"

scrape_configs:
  - job_name: 'autoformalize'
    static_configs:
      - targets: ['autoformalize:8000']
    scrape_interval: 10s
    metrics_path: /metrics
    
  - job_name: 'autoformalize-batch'
    static_configs:
      - targets: ['autoformalize-batch:8000']
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Alert Rules

```yaml
# prometheus/autoformalize_rules.yml
groups:
  - name: autoformalize.rules
    rules:
      # Success rate alerts
      - alert: LowFormalizationSuccessRate
        expr: formalization_success_rate < 0.7
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low formalization success rate"
          description: "Success rate for {{ $labels.target_system }} in {{ $labels.domain }} is {{ $value }}"
      
      # High error rate
      - alert: HighErrorRate
        expr: rate(formalizations_total{status="error"}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High formalization error rate"
          description: "Error rate is {{ $value }} errors per second"
      
      # LLM cost alerts
      - alert: HighLLMCosts
        expr: increase(llm_cost_usd_total[1h]) > 50
        for: 0m
        labels:
          severity: warning
        annotations:
          summary: "High LLM usage costs"
          description: "LLM costs exceeded $50 in the last hour"
      
      # Performance alerts
      - alert: SlowFormalizationPerformance
        expr: histogram_quantile(0.95, formalization_duration_seconds_bucket) > 60
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Slow formalization performance"
          description: "95th percentile formalization time is {{ $value }}s"
      
      # System resource alerts
      - alert: HighActiveFormalizationsCount
        expr: active_formalizations > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High number of active formalizations"
          description: "{{ $value }} active formalizations running"
```

## Grafana Dashboards

### Main Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Autoformalize Math Lab",
    "tags": ["autoformalize", "mathematics"],
    "panels": [
      {
        "title": "Formalization Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "formalization_success_rate",
            "legendFormat": "{{ target_system }} - {{ domain }}"
          }
        ]
      },
      {
        "title": "Formalizations per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(formalizations_total[5m])",
            "legendFormat": "{{ target_system }} - {{ status }}"
          }
        ]
      },
      {
        "title": "LLM Token Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(llm_tokens_used_total[5m])",
            "legendFormat": "{{ provider }} - {{ model }} - {{ type }}"
          }
        ]
      },
      {
        "title": "Verification Duration",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(proof_verification_duration_seconds_bucket[5m])",
            "legendFormat": "{{ system }}"
          }
        ]
      }
    ]
  }
}
```

## Best Practices

1. **Cardinality**: Keep label cardinality low to avoid metric explosion
2. **Naming**: Use consistent naming conventions (prefix_unit_total)
3. **Labels**: Use labels for dimensions, not for high-cardinality data
4. **Histograms**: Choose bucket boundaries based on actual data distribution
5. **Recording Rules**: Use recording rules for expensive queries
6. **Retention**: Configure appropriate retention based on storage capacity
7. **Documentation**: Document all custom metrics and their purpose