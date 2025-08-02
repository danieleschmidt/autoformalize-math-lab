#\!/bin/bash
set -e

echo "🚀 Setting up Autoformalize Math Lab development environment..."

# Update package manager
sudo apt-get update

# Install additional development tools
sudo apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Python development dependencies
pip install --upgrade pip setuptools wheel

# Install project dependencies
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
mkdir -p logs tmp data

# Set up git configuration if not already set
if [ -z "$(git config --global user.name)" ]; then
    echo "⚠️  Git user.name not configured. Please run:"
    echo "   git config --global user.name 'Your Name'"
fi

if [ -z "$(git config --global user.email)" ]; then
    echo "⚠️  Git user.email not configured. Please run:"
    echo "   git config --global user.email 'your.email@example.com'"
fi

# Install Lean 4 (if not already installed)
if \! command -v lean &> /dev/null; then
    echo "📦 Installing Lean 4..."
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- --default-toolchain leanprover/lean4:stable -y
    source ~/.profile
fi

echo "✅ Development environment setup complete\!"
echo "💡 Run 'make help' to see available commands"
