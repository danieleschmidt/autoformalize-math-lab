#!/bin/bash
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
