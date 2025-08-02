# Health Check Configuration

## Overview

Health checks are essential for monitoring service availability and automatically detecting failures in production environments.

## Health Check Endpoints

### Basic Health Check

```python
# src/autoformalize/health.py
from flask import Flask, jsonify
import time
import psutil
import subprocess

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    })

@app.route('/health/detailed')
def detailed_health_check():
    """Detailed health check with system metrics"""
    try:
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check proof assistant availability
        lean_available = check_lean_availability()
        isabelle_available = check_isabelle_availability()
        
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": (disk.used / disk.total) * 100
            },
            "services": {
                "lean4": lean_available,
                "isabelle": isabelle_available,
                "llm_api": check_llm_api()
            }
        }
        
        # Determine overall health
        if cpu_percent > 90 or memory.percent > 90:
            health_status["status"] = "degraded"
        
        if not (lean_available and isabelle_available):
            health_status["status"] = "unhealthy"
            
        return jsonify(health_status)
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 500

def check_lean_availability():
    """Check if Lean 4 is available"""
    try:
        result = subprocess.run(['lean', '--version'], 
                              capture_output=True, 
                              timeout=5)
        return result.returncode == 0
    except:
        return False

def check_isabelle_availability():
    """Check if Isabelle is available"""
    try:
        result = subprocess.run(['isabelle', 'version'], 
                              capture_output=True, 
                              timeout=5)
        return result.returncode == 0
    except:
        return False

def check_llm_api():
    """Check LLM API connectivity"""
    # Implement based on your LLM provider
    return True
```

### Kubernetes Health Checks

```yaml
# kubernetes/health-check.yaml
apiVersion: v1
kind: Pod
metadata:
  name: autoformalize-pod
spec:
  containers:
  - name: autoformalize
    image: autoformalize/math-lab:latest
    ports:
    - containerPort: 8080
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3
    readinessProbe:
      httpGet:
        path: /health/detailed
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5
      timeoutSeconds: 3
      failureThreshold: 2
```

## Docker Health Checks

```dockerfile
# Dockerfile health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

## Monitoring Checks

### External Monitoring

```bash
#!/bin/bash
# scripts/external_health_check.sh

ENDPOINT="https://your-app.com/health"
TIMEOUT=10

response=$(curl -s -w "%{http_code}" --max-time $TIMEOUT "$ENDPOINT")
http_code=${response: -3}

if [ "$http_code" = "200" ]; then
    echo "Service is healthy"
    exit 0
else
    echo "Service is unhealthy (HTTP: $http_code)"
    exit 1
fi
```

### Prometheus Health Check Metrics

```python
# Health check metrics for Prometheus
from prometheus_client import Counter, Histogram, Gauge

health_check_total = Counter('health_check_requests_total', 
                           'Total health check requests', 
                           ['endpoint', 'status'])

health_check_duration = Histogram('health_check_duration_seconds',
                                'Health check request duration')

service_availability = Gauge('service_availability',
                           'Service availability status',
                           ['service'])

@health_check_duration.time()
def perform_health_check():
    try:
        # Perform health checks
        result = check_service_health()
        health_check_total.labels(endpoint='health', status='success').inc()
        service_availability.labels(service='autoformalize').set(1)
        return result
    except Exception as e:
        health_check_total.labels(endpoint='health', status='error').inc()
        service_availability.labels(service='autoformalize').set(0)
        raise
```

## Best Practices

1. **Multiple Levels**: Implement both basic and detailed health checks
2. **Timeouts**: Set appropriate timeouts to avoid hanging checks
3. **Dependencies**: Check critical dependencies (databases, external APIs)
4. **Resource Monitoring**: Monitor CPU, memory, and disk usage
5. **Graceful Degradation**: Return degraded status before complete failure
6. **Logging**: Log health check failures for debugging
7. **Security**: Don't expose sensitive information in health responses