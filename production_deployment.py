#!/usr/bin/env python3
"""
Production Deployment Preparation
Generates production-ready configurations, checks, and deployment scripts.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

def create_production_docker_compose() -> str:
    """Create production Docker Compose configuration."""
    return """version: '3.8'

services:
  autoformalize-api:
    build:
      context: .
      dockerfile: deployment/production/Dockerfile.production
    container_name: autoformalize-api
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - WORKERS=4
      - MAX_REQUESTS=1000
      - REDIS_URL=redis://redis:6379
      - PROMETHEUS_METRICS=true
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./cache:/app/cache
    depends_on:
      - redis
      - prometheus
    networks:
      - autoformalize-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis:
    image: redis:7-alpine
    container_name: autoformalize-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    networks:
      - autoformalize-net
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: autoformalize-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deployment/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./deployment/nginx/ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - autoformalize-api
    networks:
      - autoformalize-net
    healthcheck:
      test: ["CMD", "nginx", "-t"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    container_name: autoformalize-prometheus
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    volumes:
      - ./deployment/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - autoformalize-net

  grafana:
    image: grafana/grafana:latest
    container_name: autoformalize-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./deployment/monitoring/grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus
    networks:
      - autoformalize-net

  node-exporter:
    image: prom/node-exporter:latest
    container_name: autoformalize-node-exporter
    restart: unless-stopped
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    networks:
      - autoformalize-net

volumes:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  autoformalize-net:
    driver: bridge
"""

def create_kubernetes_deployment() -> str:
    """Create Kubernetes deployment configuration."""
    return """apiVersion: apps/v1
kind: Deployment
metadata:
  name: autoformalize-api
  namespace: autoformalize
  labels:
    app: autoformalize-api
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: autoformalize-api
  template:
    metadata:
      labels:
        app: autoformalize-api
        version: v1
    spec:
      containers:
      - name: autoformalize-api
        image: autoformalize/math-lab:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: WORKERS
          value: "4"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: cache-volume
          mountPath: /app/cache
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: cache-volume
        persistentVolumeClaim:
          claimName: autoformalize-cache-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: autoformalize-logs-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: autoformalize-api-service
  namespace: autoformalize
spec:
  selector:
    app: autoformalize-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: autoformalize-config
  namespace: autoformalize
data:
  config.yaml: |
    api:
      host: "0.0.0.0"
      port: 8000
      workers: 4
    cache:
      type: "redis"
      url: "redis://redis-service:6379"
    logging:
      level: "INFO"
      format: "json"
    monitoring:
      enabled: true
      endpoint: "/metrics"

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: autoformalize-hpa
  namespace: autoformalize
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: autoformalize-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""

def create_nginx_config() -> str:
    """Create production Nginx configuration."""
    return """events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent "$http_referer" '
                   '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    # Basic settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 10M;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=formalize_limit:10m rate=2r/s;

    # Upstream backend
    upstream autoformalize_backend {
        least_conn;
        server autoformalize-api:8000;
        # Add more servers for load balancing
        # server autoformalize-api-2:8000;
        # server autoformalize-api-3:8000;
    }

    # HTTP server (redirect to HTTPS)
    server {
        listen 80;
        server_name _;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name _;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "no-referrer-when-downgrade" always;
        add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

        # API endpoints
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://autoformalize_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # Formalization endpoint (more restrictive)
        location /api/formalize {
            limit_req zone=formalize_limit burst=5 nodelay;
            
            proxy_pass http://autoformalize_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Health check
        location /health {
            proxy_pass http://autoformalize_backend;
            access_log off;
        }

        # Metrics (restrict access)
        location /metrics {
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
            
            proxy_pass http://autoformalize_backend;
        }

        # Static files (if any)
        location /static/ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
"""

def create_prometheus_config() -> str:
    """Create Prometheus monitoring configuration."""
    return """global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'autoformalize-api'
    static_configs:
      - targets: ['autoformalize-api:8000']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
"""

def create_deployment_scripts() -> Dict[str, str]:
    """Create deployment and management scripts."""
    scripts = {}
    
    # Deploy script
    scripts['deploy.sh'] = """#!/bin/bash
set -e

echo "🚀 Starting Autoformalize Math Lab Production Deployment"

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
BACKUP_DIR=${BACKUP_DIR:-./backups}
LOG_FILE="deployment_$(date +%Y%m%d_%H%M%S).log"

# Create directories
mkdir -p logs cache backups deployment/ssl

# Function to log messages
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Pre-deployment checks
log "Running pre-deployment checks..."

# Check Docker
if ! command -v docker &> /dev/null; then
    log "ERROR: Docker is not installed"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    log "ERROR: Docker Compose is not installed"
    exit 1
fi

# Check required files
required_files=(
    "docker-compose.production.yml"
    "deployment/production/Dockerfile.production"
    "deployment/nginx/nginx.conf"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        log "ERROR: Required file $file not found"
        exit 1
    fi
done

# Backup existing deployment
if [[ -f "docker-compose.production.yml" ]]; then
    log "Creating backup..."
    cp docker-compose.production.yml "$BACKUP_DIR/docker-compose.backup.$(date +%Y%m%d_%H%M%S).yml"
fi

# Build and deploy
log "Building Docker images..."
docker-compose -f docker-compose.production.yml build --no-cache

log "Starting services..."
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be ready
log "Waiting for services to be ready..."
sleep 30

# Health checks
log "Running health checks..."
services=("autoformalize-api" "redis" "nginx" "prometheus")

for service in "${services[@]}"; do
    if docker-compose -f docker-compose.production.yml ps "$service" | grep -q "Up"; then
        log "✅ $service is running"
    else
        log "❌ $service is not running"
        exit 1
    fi
done

# API health check
if curl -f http://localhost/health > /dev/null 2>&1; then
    log "✅ API health check passed"
else
    log "❌ API health check failed"
    exit 1
fi

log "🎉 Deployment completed successfully!"
log "📊 Grafana: http://localhost:3000 (admin/admin123)"
log "📈 Prometheus: http://localhost:9090"
log "🔍 API: http://localhost/api/docs"
"""

    # Health check script
    scripts['healthcheck.sh'] = """#!/bin/bash
set -e

echo "🏥 Autoformalize Health Check"
echo "=============================="

# Check Docker containers
echo "📦 Docker Containers:"
docker-compose -f docker-compose.production.yml ps

# Check API health
echo ""
echo "🔍 API Health:"
if curl -f -s http://localhost/health | jq .; then
    echo "✅ API is healthy"
else
    echo "❌ API health check failed"
fi

# Check Redis
echo ""
echo "🔴 Redis:"
if docker exec autoformalize-redis redis-cli ping | grep -q PONG; then
    echo "✅ Redis is responding"
else
    echo "❌ Redis is not responding"
fi

# Check Nginx
echo ""
echo "🌐 Nginx:"
if curl -f -s -o /dev/null http://localhost; then
    echo "✅ Nginx is serving"
else
    echo "❌ Nginx is not responding"
fi

# System resources
echo ""
echo "💻 System Resources:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')"
echo "Memory: $(free -h | awk 'NR==2{printf "%.1f%%\\n", $3*100/$2}')"
echo "Disk: $(df -h / | awk 'NR==2{print $5}')"

# Container logs (last 10 lines)
echo ""
echo "📋 Recent Container Logs:"
echo "--- API Logs ---"
docker-compose -f docker-compose.production.yml logs --tail=5 autoformalize-api

echo ""
echo "--- Nginx Logs ---"
docker-compose -f docker-compose.production.yml logs --tail=5 nginx
"""

    # Backup script
    scripts['backup.sh'] = """#!/bin/bash
set -e

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Creating backup in $BACKUP_DIR"

# Backup Redis data
echo "Backing up Redis data..."
docker exec autoformalize-redis redis-cli BGSAVE
sleep 5
docker cp autoformalize-redis:/data/dump.rdb "$BACKUP_DIR/"

# Backup configuration files
echo "Backing up configuration..."
cp docker-compose.production.yml "$BACKUP_DIR/"
cp -r deployment "$BACKUP_DIR/"

# Backup logs
echo "Backing up logs..."
cp -r logs "$BACKUP_DIR/" 2>/dev/null || true

# Create manifest
echo "Creating backup manifest..."
cat > "$BACKUP_DIR/manifest.txt" << EOF
Backup created: $(date)
Redis data: dump.rdb
Configuration: docker-compose.production.yml, deployment/
Logs: logs/
EOF

echo "✅ Backup completed: $BACKUP_DIR"
"""

    # Rollback script
    scripts['rollback.sh'] = """#!/bin/bash
set -e

if [[ -z "$1" ]]; then
    echo "Usage: $0 <backup_directory>"
    echo "Available backups:"
    ls -la backups/
    exit 1
fi

BACKUP_DIR="$1"

if [[ ! -d "$BACKUP_DIR" ]]; then
    echo "ERROR: Backup directory $BACKUP_DIR not found"
    exit 1
fi

echo "🔄 Rolling back from $BACKUP_DIR"

# Stop current services
echo "Stopping current services..."
docker-compose -f docker-compose.production.yml down

# Restore configuration
echo "Restoring configuration..."
cp "$BACKUP_DIR/docker-compose.production.yml" .
cp -r "$BACKUP_DIR/deployment" .

# Restore Redis data
echo "Restoring Redis data..."
docker-compose -f docker-compose.production.yml up -d redis
sleep 10
docker cp "$BACKUP_DIR/dump.rdb" autoformalize-redis:/data/
docker-compose -f docker-compose.production.yml restart redis

# Start all services
echo "Starting services..."
docker-compose -f docker-compose.production.yml up -d

echo "✅ Rollback completed"
"""

    return scripts

def create_environment_config() -> str:
    """Create environment configuration template."""
    return """# Production Environment Configuration
# Copy this file to .env.production and customize

# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4
MAX_REQUESTS=1000
TIMEOUT=30

# Database & Cache
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600

# LLM Configuration
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
LLM_TIMEOUT=60
MAX_RETRIES=3

# Security
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
CORS_ORIGINS=https://your-domain.com
CSRF_PROTECTION=true
RATE_LIMIT_PER_MINUTE=60

# Monitoring
PROMETHEUS_METRICS=true
GRAFANA_ADMIN_PASSWORD=your-secure-password
ENABLE_PROFILING=false

# SSL/TLS
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem

# Backup
BACKUP_RETENTION_DAYS=30
AUTO_BACKUP_ENABLED=true
BACKUP_SCHEDULE=0 2 * * *  # Daily at 2 AM

# Scaling
AUTO_SCALE_ENABLED=true
MIN_REPLICAS=3
MAX_REPLICAS=20
TARGET_CPU_UTILIZATION=70
TARGET_MEMORY_UTILIZATION=80
"""

def setup_production_deployment():
    """Set up production deployment files and configurations."""
    print("🚀 PRODUCTION DEPLOYMENT SETUP")
    print("=" * 50)
    
    repo_root = Path(__file__).parent
    
    # Create deployment directories
    deployment_dirs = [
        "deployment/production",
        "deployment/nginx",
        "deployment/monitoring",
        "deployment/ssl",
        "logs",
        "cache", 
        "backups"
    ]
    
    for dir_path in deployment_dirs:
        (repo_root / dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
    
    # Create deployment files
    files_to_create = {
        "docker-compose.production.yml": create_production_docker_compose(),
        "deployment/kubernetes/deployment.yaml": create_kubernetes_deployment(),
        "deployment/nginx/nginx.conf": create_nginx_config(),
        "deployment/monitoring/prometheus.yml": create_prometheus_config(),
        ".env.production.template": create_environment_config(),
    }
    
    for file_path, content in files_to_create.items():
        full_path = repo_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"📄 Created file: {file_path}")
    
    # Create deployment scripts
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    
    scripts = create_deployment_scripts()
    for script_name, script_content in scripts.items():
        script_path = scripts_dir / script_name
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)  # Make executable
        print(f"🔧 Created script: scripts/{script_name}")
    
    # Create production Dockerfile
    production_dockerfile = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY pyproject.toml .

# Install the application
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "autoformalize.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
"""
    
    dockerfile_path = repo_root / "deployment/production/Dockerfile.production"
    with open(dockerfile_path, 'w') as f:
        f.write(production_dockerfile)
    print(f"🐳 Created production Dockerfile")
    
    # Create deployment summary
    summary = {
        "deployment_type": "production",
        "created_files": list(files_to_create.keys()) + [f"scripts/{name}" for name in scripts.keys()],
        "directories_created": deployment_dirs,
        "services": [
            "autoformalize-api",
            "redis",
            "nginx", 
            "prometheus",
            "grafana",
            "node-exporter"
        ],
        "ports": {
            "HTTP": 80,
            "HTTPS": 443,
            "API": 8000,
            "Redis": 6379,
            "Prometheus": 9090,
            "Grafana": 3000,
            "Node Exporter": 9100
        },
        "next_steps": [
            "1. Copy .env.production.template to .env.production and customize",
            "2. Generate SSL certificates and place in deployment/ssl/",
            "3. Review and customize nginx.conf for your domain",
            "4. Run: chmod +x scripts/*.sh",
            "5. Deploy with: ./scripts/deploy.sh",
            "6. Monitor with: ./scripts/healthcheck.sh"
        ]
    }
    
    summary_path = repo_root / "deployment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ PRODUCTION DEPLOYMENT SETUP COMPLETE")
    print(f"📊 Summary saved to: deployment_summary.json")
    print(f"\n🚀 Next Steps:")
    for step in summary["next_steps"]:
        print(f"   {step}")
    
    print(f"\n🔍 Services will be available at:")
    for service, port in summary["ports"].items():
        if port in [80, 443]:
            print(f"   {service}: http{'s' if port == 443 else ''}://localhost")
        else:
            print(f"   {service}: http://localhost:{port}")
    
    return summary

if __name__ == "__main__":
    try:
        setup_production_deployment()
        print("\n🎉 Production deployment preparation completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Production deployment setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)