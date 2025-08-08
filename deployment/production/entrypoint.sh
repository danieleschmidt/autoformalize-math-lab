#!/bin/bash
set -e

# Production entrypoint script for autoformalize-math-lab

# Environment variables with defaults
export ENVIRONMENT=${ENVIRONMENT:-production}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export WORKERS=${WORKERS:-4}
export MAX_REQUESTS=${MAX_REQUESTS:-1000}
export MAX_REQUESTS_JITTER=${MAX_REQUESTS_JITTER:-100}
export TIMEOUT=${TIMEOUT:-120}
export KEEP_ALIVE=${KEEP_ALIVE:-5}

# Wait for dependent services
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local timeout=${4:-30}
    
    echo "Waiting for $service at $host:$port..."
    
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "$service is ready!"
            return 0
        fi
        echo "Waiting for $service... ($i/$timeout)"
        sleep 1
    done
    
    echo "ERROR: $service at $host:$port is not available after $timeout seconds"
    return 1
}

# Database migration function
run_migrations() {
    echo "Running database migrations..."
    python -m alembic upgrade head || {
        echo "ERROR: Database migration failed"
        exit 1
    }
}

# Cache warming function
warm_cache() {
    echo "Warming application cache..."
    python -c "
import sys
sys.path.insert(0, 'src')
from autoformalize.utils.caching import CacheManager
cache = CacheManager()
cache.warm_cache()
print('Cache warmed successfully')
" || echo "WARNING: Cache warming failed, continuing anyway"
}

# Health check function
health_check() {
    echo "Running health check..."
    python healthcheck.py || {
        echo "ERROR: Health check failed"
        exit 1
    }
}

# Signal handlers for graceful shutdown
cleanup() {
    echo "Received shutdown signal, performing cleanup..."
    
    # Save any pending cache data
    python -c "
import sys
sys.path.insert(0, 'src')
from autoformalize.utils.caching import CacheManager
cache = CacheManager()
cache.flush()
print('Cache flushed')
" 2>/dev/null || true

    # Graceful shutdown
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
    
    echo "Cleanup completed"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Main execution
main() {
    echo "Starting autoformalize-math-lab in $ENVIRONMENT mode..."
    
    # Verify Python path
    export PYTHONPATH="/app/src:$PYTHONPATH"
    
    # Wait for Redis
    if [ -n "$REDIS_URL" ]; then
        redis_host=$(echo $REDIS_URL | sed -E 's|redis://([^:]+):([0-9]+)/.*|\1|')
        redis_port=$(echo $REDIS_URL | sed -E 's|redis://([^:]+):([0-9]+)/.*|\2|')
        wait_for_service "$redis_host" "$redis_port" "Redis"
    fi
    
    # Wait for PostgreSQL
    if [ -n "$POSTGRES_URL" ]; then
        postgres_host=$(echo $POSTGRES_URL | sed -E 's|postgresql://[^@]+@([^:]+):([0-9]+)/.*|\1|')
        postgres_port=$(echo $POSTGRES_URL | sed -E 's|postgresql://[^@]+@([^:]+):([0-9]+)/.*|\2|')
        wait_for_service "$postgres_host" "$postgres_port" "PostgreSQL"
    fi
    
    # Handle different startup modes
    case "${1:-server}" in
        "server")
            echo "Starting API server..."
            run_migrations
            warm_cache
            
            # Start Gunicorn with proper configuration
            exec gunicorn \
                --bind 0.0.0.0:8000 \
                --workers $WORKERS \
                --worker-class uvicorn.workers.UvicornWorker \
                --max-requests $MAX_REQUESTS \
                --max-requests-jitter $MAX_REQUESTS_JITTER \
                --timeout $TIMEOUT \
                --keep-alive $KEEP_ALIVE \
                --access-logfile - \
                --error-logfile - \
                --log-level $LOG_LEVEL \
                --preload \
                --enable-stdio-inheritance \
                autoformalize.api.server:app &
            
            child=$!
            wait "$child"
            ;;
            
        "worker")
            echo "Starting worker process..."
            
            # Start distributed worker
            exec python -m autoformalize.scaling.worker &
            child=$!
            wait "$child"
            ;;
            
        "scheduler")
            echo "Starting scheduler process..."
            
            # Start task scheduler
            exec python -m autoformalize.utils.scheduler &
            child=$!
            wait "$child"
            ;;
            
        "migrate")
            echo "Running migrations only..."
            run_migrations
            echo "Migrations completed"
            ;;
            
        "shell")
            echo "Starting interactive shell..."
            exec python -i -c "
import sys
sys.path.insert(0, 'src')
from autoformalize import *
print('Autoformalize shell ready')
"
            ;;
            
        "test")
            echo "Running tests..."
            exec python -m pytest tests/ -v
            ;;
            
        "health")
            health_check
            echo "Health check passed"
            ;;
            
        *)
            echo "Starting custom command: $@"
            exec "$@" &
            child=$!
            wait "$child"
            ;;
    esac
}

# Ensure we're in the correct directory
cd /app

# Install netcat for service waiting
command -v nc >/dev/null 2>&1 || {
    echo "Installing netcat..."
    apt-get update && apt-get install -y netcat && rm -rf /var/lib/apt/lists/*
}

# Run main function
main "$@"