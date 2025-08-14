#!/bin/bash
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
echo "Memory: $(free -h | awk 'NR==2{printf "%.1f%%\n", $3*100/$2}')"
echo "Disk: $(df -h / | awk 'NR==2{print $5}')"

# Container logs (last 10 lines)
echo ""
echo "📋 Recent Container Logs:"
echo "--- API Logs ---"
docker-compose -f docker-compose.production.yml logs --tail=5 autoformalize-api

echo ""
echo "--- Nginx Logs ---"
docker-compose -f docker-compose.production.yml logs --tail=5 nginx
