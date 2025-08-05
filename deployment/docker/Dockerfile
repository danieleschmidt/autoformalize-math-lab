# Multi-stage production Dockerfile for Autoformalize
FROM python:3.12-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Add metadata
LABEL org.opencontainers.image.title="Autoformalize" \
      org.opencontainers.image.description="Mathematical formalization service using LLMs and formal proof assistants" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.source="https://github.com/danieleschmidt/quantum-inspired-task-planner"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY README.md CHANGELOG.md LICENSE ./

# Install the package
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.12-slim as production

# Create non-root user for security
RUN groupadd -r autoformalize && useradd -r -g autoformalize autoformalize

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src /app/src

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/cache && \
    chown -R autoformalize:autoformalize /app

# Copy configuration files
COPY deployment/config/ ./config/
COPY deployment/scripts/ ./scripts/

# Make scripts executable
RUN chmod +x ./scripts/*.sh

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AUTOFORMALIZE_ENV=production \
    AUTOFORMALIZE_LOG_LEVEL=INFO \
    AUTOFORMALIZE_DATA_DIR=/app/data \
    AUTOFORMALIZE_CACHE_DIR=/app/cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/status || exit 1

# Switch to non-root user
USER autoformalize

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "autoformalize.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]