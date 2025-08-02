#\!/bin/bash
set -e

echo "🔄 Running post-start setup..."

# Activate conda environment if available
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

# Start development services in background if docker-compose is available
if [ -f "docker-compose.yml" ] && command -v docker-compose &> /dev/null; then
    echo "🐳 Starting development services..."
    docker-compose up -d redis postgres || echo "ℹ️  Some services might not be available"
fi

echo "✅ Post-start setup complete\!"
