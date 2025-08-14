#!/bin/bash
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
