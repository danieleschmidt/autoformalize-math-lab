#!/bin/bash
# Build script for autoformalize-math-lab
# Provides standardized build commands for CI/CD and local development

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
DIST_DIR="$PROJECT_ROOT/dist"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
PYTHON=${PYTHON:-python3}
PIP=${PIP:-pip}
BUILD_TYPE=${BUILD_TYPE:-development}
SKIP_TESTS=${SKIP_TESTS:-false}
SKIP_LINT=${SKIP_LINT:-false}
DOCKER_BUILD=${DOCKER_BUILD:-false}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Build script for autoformalize-math-lab

COMMANDS:
    clean           Clean build artifacts
    install         Install dependencies
    lint            Run code quality checks  
    test            Run test suite
    build           Build Python packages
    docker          Build Docker images
    release         Build release packages
    all             Run complete CI/CD pipeline

OPTIONS:
    -h, --help      Show this help message
    -t, --type      Build type: development|production (default: development)
    --skip-tests    Skip running tests
    --skip-lint     Skip linting and code quality checks
    --docker        Build Docker images
    --python PATH   Python executable to use (default: python3)

ENVIRONMENT VARIABLES:
    BUILD_TYPE      Build type (development|production)
    SKIP_TESTS      Skip tests (true|false)
    SKIP_LINT       Skip linting (true|false)
    DOCKER_BUILD    Build Docker images (true|false)
    PYTHON          Python executable path
    PIP             Pip executable path

Examples:
    $0 clean
    $0 install
    $0 --type production build
    $0 --docker all
    BUILD_TYPE=production $0 build
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-lint)
            SKIP_LINT=true
            shift
            ;;
        --docker)
            DOCKER_BUILD=true
            shift
            ;;
        --python)
            PYTHON="$2"
            shift 2
            ;;
        -*)
            log_error "Unknown option: $1"
            exit 1
            ;;
        *)
            COMMAND="$1"
            shift
            ;;
    esac
done

# Validate build type
if [[ "$BUILD_TYPE" != "development" && "$BUILD_TYPE" != "production" ]]; then
    log_error "Invalid build type: $BUILD_TYPE. Must be 'development' or 'production'"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Build functions
clean_build() {
    log_info "Cleaning build artifacts..."
    
    # Remove Python build artifacts
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    rm -rf .pytest_cache/
    rm -rf .mypy_cache/
    rm -rf .coverage
    rm -rf htmlcov/
    rm -rf .tox/
    
    # Remove Python cache files
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.orig" -delete
    find . -type f -name "*.rej" -delete
    
    # Remove temporary files
    rm -rf temp/
    rm -rf tmp/
    rm -rf .tmp/
    
    log_success "Build artifacts cleaned"
}

install_dependencies() {
    log_info "Installing dependencies for $BUILD_TYPE build..."
    
    # Upgrade pip
    $PIP install --upgrade pip setuptools wheel
    
    if [[ "$BUILD_TYPE" == "development" ]]; then
        # Install with development dependencies
        $PIP install -e ".[dev,docs]"
        
        # Install pre-commit hooks
        if command -v pre-commit >/dev/null 2>&1; then
            pre-commit install
            pre-commit install --hook-type commit-msg
        else
            log_warning "pre-commit not found, skipping hook installation"
        fi
    else
        # Production installation
        $PIP install -e .
    fi
    
    log_success "Dependencies installed"
}

run_lint() {
    if [[ "$SKIP_LINT" == "true" ]]; then
        log_info "Skipping code quality checks"
        return 0
    fi
    
    log_info "Running code quality checks..."
    
    # Code formatting check
    if command -v black >/dev/null 2>&1; then
        log_info "Checking code formatting with black..."
        black --check src/ tests/ || {
            log_error "Code formatting check failed. Run 'black src/ tests/' to fix"
            return 1
        }
    fi
    
    # Import sorting check
    if command -v isort >/dev/null 2>&1; then
        log_info "Checking import sorting with isort..."
        isort --check-only src/ tests/ || {
            log_error "Import sorting check failed. Run 'isort src/ tests/' to fix"
            return 1
        }
    fi
    
    # Linting
    if command -v flake8 >/dev/null 2>&1; then
        log_info "Running flake8 linting..."
        flake8 src/ tests/
    elif command -v ruff >/dev/null 2>&1; then
        log_info "Running ruff linting..."
        ruff check src/ tests/
    else
        log_warning "No linter found (flake8 or ruff)"
    fi
    
    # Type checking
    if command -v mypy >/dev/null 2>&1; then
        log_info "Running type checking with mypy..."
        mypy src/
    else
        log_warning "mypy not found, skipping type checking"
    fi
    
    # Security scanning
    if command -v bandit >/dev/null 2>&1; then
        log_info "Running security scan with bandit..."
        bandit -r src/ -f json -o security-report.json || true
        bandit -r src/
    else
        log_warning "bandit not found, skipping security scan"
    fi
    
    log_success "Code quality checks passed"
}

run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_info "Skipping tests"
        return 0
    fi
    
    log_info "Running test suite..."
    
    if command -v pytest >/dev/null 2>&1; then
        # Run tests with coverage
        pytest tests/ -v \
            --cov=src/autoformalize \
            --cov-report=term-missing \
            --cov-report=html:htmlcov \
            --cov-report=xml:coverage.xml \
            --tb=short \
            --durations=10
    else
        log_error "pytest not found"
        return 1
    fi
    
    log_success "Tests passed"
}

build_packages() {
    log_info "Building Python packages..."
    
    # Clean previous builds
    rm -rf build/ dist/
    
    # Build packages
    $PYTHON -m build
    
    # Verify build
    if [[ -d "dist" && -n "$(ls -A dist/)" ]]; then
        log_success "Packages built successfully:"
        ls -la dist/
    else
        log_error "Package build failed"
        return 1
    fi
}

build_docker() {
    if [[ "$DOCKER_BUILD" != "true" ]]; then
        return 0
    fi
    
    log_info "Building Docker images..."
    
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker not found"
        return 1
    fi
    
    # Build main image
    docker build -t autoformalize-math-lab:latest .
    
    # Build development image
    docker build --target development -t autoformalize-math-lab:dev .
    
    # Build GPU image if CUDA is available
    if command -v nvidia-smi >/dev/null 2>&1; then
        log_info "Building GPU-enabled image..."
        docker build --target gpu-base -t autoformalize-math-lab:gpu .
    fi
    
    log_success "Docker images built"
}

build_release() {
    log_info "Building release packages..."
    
    # Set production build type
    BUILD_TYPE=production
    
    # Clean everything first
    clean_build
    
    # Install production dependencies
    install_dependencies
    
    # Run quality checks
    run_lint
    
    # Run tests
    run_tests
    
    # Build packages
    build_packages
    
    # Build Docker images for release
    DOCKER_BUILD=true
    build_docker
    
    log_success "Release build completed"
}

run_all() {
    log_info "Running complete CI/CD pipeline..."
    
    clean_build
    install_dependencies
    run_lint
    run_tests
    build_packages
    build_docker
    
    log_success "Complete build pipeline finished"
}

# Main command handling
case "${COMMAND:-help}" in
    clean)
        clean_build
        ;;
    install)
        install_dependencies
        ;;
    lint)
        run_lint
        ;;
    test)
        run_tests
        ;;
    build)
        build_packages
        ;;
    docker)
        DOCKER_BUILD=true
        build_docker
        ;;
    release)
        build_release
        ;;
    all)
        run_all
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: ${COMMAND}"
        echo "Use '$0 --help' for usage information"
        exit 1
        ;;
esac