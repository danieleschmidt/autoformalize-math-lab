#!/bin/bash
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
