# Development container for autoformalize-math-lab
# Based on the main Dockerfile but optimized for development

FROM mcr.microsoft.com/vscode/devcontainers/python:3.11

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    # LaTeX support
    texlive-latex-base \
    texlive-latex-extra \
    texlive-math-extra \
    texlive-fonts-recommended \
    # Development tools
    vim \
    nano \
    htop \
    tree \
    jq \
    # Graphics and media
    imagemagick \
    graphviz \
    # Networking tools
    netcat \
    telnet \
    # Additional utilities
    zip \
    unzip \
    less \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (vscode user already exists in base image)
USER vscode
ENV HOME=/home/vscode
WORKDIR $HOME

# Install Lean 4
ENV ELAN_HOME=$HOME/.elan
ENV PATH="$ELAN_HOME/bin:$PATH"
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain leanprover/lean4:stable

# Install additional development tools
RUN pip install --user --no-cache-dir \
    # Development tools
    ipython \
    jupyter \
    jupyterlab \
    notebook \
    # Debugging
    debugpy \
    # Documentation
    sphinx \
    sphinx-rtd-theme \
    # Performance profiling
    line-profiler \
    memory-profiler \
    # Code formatting and analysis
    pre-commit \
    # Additional utilities
    rich \
    typer

# Configure git (will be overridden by user config)
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false

# Set up shell environment
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc \
    && echo 'export PYTHONPATH="/workspace/src:$PYTHONPATH"' >> ~/.bashrc \
    && echo 'alias ll="ls -la"' >> ~/.bashrc \
    && echo 'alias la="ls -A"' >> ~/.bashrc \
    && echo 'alias l="ls -CF"' >> ~/.bashrc \
    && echo 'alias ..="cd .."' >> ~/.bashrc \
    && echo 'alias ...="cd ../.."' >> ~/.bashrc \
    && echo 'alias grep="grep --color=auto"' >> ~/.bashrc \
    && echo 'alias fgrep="fgrep --color=auto"' >> ~/.bashrc \
    && echo 'alias egrep="egrep --color=auto"' >> ~/.bashrc

# Create common directories
RUN mkdir -p \
    $HOME/.cache \
    $HOME/.local \
    $HOME/.jupyter \
    $HOME/workspace

# Configure Jupyter
RUN jupyter --generate-config \
    && echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

# Install Oh My Bash for better shell experience
RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)" --unattended \
    && sed -i 's/OSH_THEME="font"/OSH_THEME="powerline"/' ~/.bashrc

# Create a startup script
RUN echo '#!/bin/bash' > ~/.startup.sh \
    && echo 'echo "🔬 autoformalize-math-lab Development Environment"' >> ~/.startup.sh \
    && echo 'echo "Python: $(python --version)"' >> ~/.startup.sh \
    && echo 'echo "Lean: $(lean --version 2>/dev/null || echo "Not installed")"' >> ~/.startup.sh \
    && echo 'echo "Working directory: $(pwd)"' >> ~/.startup.sh \
    && echo 'echo ""' >> ~/.startup.sh \
    && echo 'echo "Quick commands:"' >> ~/.startup.sh \
    && echo 'echo "  make test     - Run tests"' >> ~/.startup.sh \
    && echo 'echo "  make docs     - Build documentation"' >> ~/.startup.sh \
    && echo 'echo "  make format   - Format code"' >> ~/.startup.sh \
    && echo 'echo "  jupyter lab   - Start Jupyter Lab"' >> ~/.startup.sh \
    && echo 'echo ""' >> ~/.startup.sh \
    && chmod +x ~/.startup.sh

# Set up working directory
WORKDIR /workspace

# Set default command
CMD ["/bin/bash", "-c", "~/.startup.sh && /bin/bash"]