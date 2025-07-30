# Multi-stage Dockerfile for autoformalize-math-lab
# Optimized for mathematical formalization workloads

# Stage 1: Base Python environment with system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies for mathematical software
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    unzip \
    ca-certificates \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-math-extra \
    texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 autoformalize && \
    useradd --uid 1000 --gid autoformalize --shell /bin/bash --create-home autoformalize

# Set working directory
WORKDIR /app

# Stage 2: Proof assistant installations
FROM base as proof-assistants

# Install Lean 4
ENV ELAN_HOME=/home/autoformalize/.elan
ENV PATH="$ELAN_HOME/bin:$PATH"

USER autoformalize
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain leanprover/lean4:stable
USER root

# Install Isabelle/HOL (placeholder - adjust version as needed)
# RUN wget https://isabelle.in.tum.de/dist/Isabelle2023_Linux.tar.gz && \
#     tar -xzf Isabelle2023_Linux.tar.gz -C /opt/ && \
#     ln -s /opt/Isabelle2023/bin/isabelle /usr/local/bin/isabelle && \
#     rm Isabelle2023_Linux.tar.gz

# Install Coq (placeholder)
# RUN apt-get update && apt-get install -y coq && rm -rf /var/lib/apt/lists/*

# Stage 3: Python dependencies
FROM proof-assistants as python-deps

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Stage 4: Development environment (optional)
FROM python-deps as development

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev,docs]"

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    less \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set up development environment
USER autoformalize
RUN echo 'alias ll="ls -la"' >> ~/.bashrc && \
    echo 'export PATH="/home/autoformalize/.local/bin:$PATH"' >> ~/.bashrc

# Stage 5: Production environment
FROM python-deps as production

# Copy application code
COPY --chown=autoformalize:autoformalize src/ ./src/
COPY --chown=autoformalize:autoformalize README.md LICENSE ./

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/cache && \
    chown -R autoformalize:autoformalize /app/data /app/outputs /app/cache

# Switch to non-root user
USER autoformalize

# Set up environment
ENV AUTOFORMALIZE_CACHE_DIR=/app/cache \
    AUTOFORMALIZE_OUTPUT_DIR=/app/outputs \
    AUTOFORMALIZE_DATA_DIR=/app/data

# Expose port for web interface (if implemented)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import autoformalize; print('OK')" || exit 1

# Default command
CMD ["autoformalize", "--help"]

# Stage 6: GPU-enabled version (for ML workloads)
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu-base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Continue with similar setup as production stage...
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -e .

# Multi-architecture support
# This Dockerfile supports both amd64 and arm64 architectures