# Deployment Guide

This guide covers various deployment options for Autoformalize Math Lab.

## Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/yourusername/autoformalize-math-lab.git
cd autoformalize-math-lab

# Using Makefile
make dev-setup

# Or manually
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Docker Development

```bash
# Start development environment
docker-compose --profile dev up -d

# Enter development container
docker-compose exec autoformalize-dev bash

# Or use devcontainer in VS Code
code .  # Open in VS Code and reopen in container
```

## Production Deployment

### Docker Compose (Recommended)

```bash
# Basic production setup
docker-compose up -d

# With monitoring and caching
docker-compose --profile production --profile monitoring --profile cache up -d

# Check status
docker-compose ps
docker-compose logs autoformalize
```

### Kubernetes

#### Prerequisites

- Kubernetes cluster (v1.20+)
- kubectl configured
- Helm 3.x (optional but recommended)

#### Using Helm Chart

```bash
# Add Helm repository (when available)
helm repo add autoformalize https://charts.autoformalize.org
helm repo update

# Install with default values
helm install autoformalize autoformalize/autoformalize-math-lab

# With custom values
helm install autoformalize autoformalize/autoformalize-math-lab -f values.yaml
```

#### Manual Kubernetes Deployment

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: autoformalize
---
# kubernetes/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: autoformalize-config
  namespace: autoformalize
data:
  AUTOFORMALIZE_LOG_LEVEL: "INFO"
  AUTOFORMALIZE_CACHE_DIR: "/app/cache"
---
# kubernetes/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: autoformalize-secrets
  namespace: autoformalize
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-key>
  ANTHROPIC_API_KEY: <base64-encoded-key>
---
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autoformalize
  namespace: autoformalize
spec:
  replicas: 2
  selector:
    matchLabels:
      app: autoformalize
  template:
    metadata:
      labels:
        app: autoformalize
    spec:
      containers:
      - name: autoformalize
        image: autoformalize/autoformalize-math-lab:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: autoformalize-config
        - secretRef:
            name: autoformalize-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: cache
          mountPath: /app/cache
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: autoformalize-cache-pvc
---
# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: autoformalize-service
  namespace: autoformalize
spec:
  selector:
    app: autoformalize
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
# kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: autoformalize-ingress
  namespace: autoformalize
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: autoformalize.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: autoformalize-service
            port:
              number: 80
```

Apply the manifests:

```bash
kubectl apply -f kubernetes/
```

### Cloud Platforms

#### AWS ECS

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name autoformalize-cluster

# Create task definition
aws ecs register-task-definition --cli-input-json file://aws-task-definition.json

# Create service
aws ecs create-service \
  --cluster autoformalize-cluster \
  --service-name autoformalize-service \
  --task-definition autoformalize:1 \
  --desired-count 2
```

#### Google Cloud Run

```bash
# Build and push image to GCR
gcloud builds submit --tag gcr.io/PROJECT-ID/autoformalize-math-lab

# Deploy to Cloud Run
gcloud run deploy autoformalize \
  --image gcr.io/PROJECT-ID/autoformalize-math-lab \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars AUTOFORMALIZE_LOG_LEVEL=INFO
```

#### Azure Container Instances

```bash
# Create resource group
az group create --name autoformalize-rg --location eastus

# Deploy container
az container create \
  --resource-group autoformalize-rg \
  --name autoformalize \
  --image autoformalize/autoformalize-math-lab:latest \
  --dns-name-label autoformalize-aci \
  --ports 8000
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `AUTOFORMALIZE_LOG_LEVEL` | Logging level | `INFO` | No |
| `OPENAI_API_KEY` | OpenAI API key | - | Yes |
| `ANTHROPIC_API_KEY` | Anthropic API key | - | No |
| `DATABASE_URL` | Database connection string | - | No |
| `REDIS_URL` | Redis connection string | - | No |

### Configuration Files

Create a `.env` file for local development:

```bash
# Copy example configuration
cp .env.example .env

# Edit with your values
vim .env
```

### Database Setup

#### PostgreSQL (Recommended)

```bash
# Using Docker
docker run -d \
  --name autoformalize-postgres \
  -e POSTGRES_DB=autoformalize \
  -e POSTGRES_USER=autoformalize \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 \
  postgres:15-alpine

# Initialize database
python scripts/init_database.py
```

#### SQLite (Development)

```bash
# SQLite is used by default for development
# Database file will be created automatically
export DATABASE_URL=sqlite:///autoformalize.db
```

### Caching Setup

#### Redis (Recommended)

```bash
# Using Docker
docker run -d \
  --name autoformalize-redis \
  -p 6379:6379 \
  redis:7-alpine

# Configure application
export REDIS_URL=redis://localhost:6379/0
```

#### File-based Cache

```bash
# File-based caching (default)
export AUTOFORMALIZE_CACHE_DIR=./cache
mkdir -p cache
```

## Monitoring and Observability

### Prometheus + Grafana

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access Grafana at http://localhost:3000
# Default credentials: admin/admin
```

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/status

# Metrics
curl http://localhost:8000/metrics
```

### Logging

```bash
# View logs
docker-compose logs -f autoformalize

# In Kubernetes
kubectl logs -f deployment/autoformalize -n autoformalize

# Log aggregation with ELK stack
docker-compose --profile logging up -d
```

## Scaling and Performance

### Horizontal Scaling

```bash
# Docker Compose
docker-compose up -d --scale autoformalize=3

# Kubernetes
kubectl scale deployment autoformalize --replicas=5 -n autoformalize
```

### Load Balancing

```nginx
# Nginx configuration
upstream autoformalize_backend {
    server autoformalize-1:8000;
    server autoformalize-2:8000;
    server autoformalize-3:8000;
    keepalive 32;
}

server {
    listen 80;
    location / {
        proxy_pass http://autoformalize_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Performance Tuning

#### Memory Optimization

```bash
# Increase memory limits
export AUTOFORMALIZE_MAX_MEMORY=4096
export AUTOFORMALIZE_CACHE_SIZE=1024
```

#### CPU Optimization

```bash
# Set worker processes
export AUTOFORMALIZE_WORKERS=4
export AUTOFORMALIZE_THREADS=2
```

#### Database Optimization

```sql
-- PostgreSQL optimizations
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
SELECT pg_reload_conf();
```

## Security

### SSL/TLS Configuration

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  nginx:
    volumes:
      - ./ssl:/etc/nginx/ssl:ro
    environment:
      - SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
      - SSL_KEY_PATH=/etc/nginx/ssl/key.pem
```

### API Security

```bash
# Generate secure API keys
export AUTOFORMALIZE_SECRET_KEY=$(openssl rand -hex 32)
export AUTOFORMALIZE_JWT_SECRET=$(openssl rand -hex 32)
```

### Network Security

```yaml
# docker-compose security
networks:
  autoformalize-network:
    driver: bridge
    internal: true  # Isolate from external networks
```

### Container Security

```dockerfile
# Use non-root user
USER autoformalize:autoformalize

# Read-only filesystem
--read-only --tmpfs /tmp --tmpfs /var/tmp
```

## Backup and Recovery

### Database Backup

```bash
# PostgreSQL backup
pg_dump -h localhost -U autoformalize -d autoformalize > backup.sql

# Automated backups
0 2 * * * /usr/bin/docker exec autoformalize-postgres pg_dump -U autoformalize autoformalize > /backups/autoformalize-$(date +\%Y\%m\%d).sql
```

### Application Data Backup

```bash
# Backup cache and logs
tar -czf autoformalize-data-$(date +%Y%m%d).tar.gz \
  cache/ logs/ outputs/

# Restore from backup
tar -xzf autoformalize-data-20231201.tar.gz
```

## Troubleshooting

### Common Issues

#### Connection Errors

```bash
# Check service status
docker-compose ps
systemctl status autoformalize

# Check network connectivity
curl -v http://localhost:8000/health
ping redis
```

#### Memory Issues

```bash
# Monitor memory usage
docker stats
kubectl top pods -n autoformalize

# Increase memory limits
docker-compose up -d --scale autoformalize=2
```

#### API Rate Limits

```bash
# Check rate limit status
curl -I http://localhost:8000/api/formalize

# Configure rate limits
export AUTOFORMALIZE_RATE_LIMIT=100
```

### Debug Mode

```bash
# Enable debug logging
export AUTOFORMALIZE_LOG_LEVEL=DEBUG
export AUTOFORMALIZE_DEBUG=true

# Run with debugger
python -m pdb -m autoformalize.cli
```

### Support and Maintenance

```bash
# Update to latest version
docker-compose pull
docker-compose up -d

# Health check script
./scripts/health_check.sh

# Maintenance mode
touch maintenance.lock  # Enables maintenance mode
```

For additional support, see:
- [Troubleshooting Guide](troubleshooting.md)
- [FAQ](faq.md)
- [GitHub Issues](https://github.com/yourusername/autoformalize-math-lab/issues)