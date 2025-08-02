#!/bin/bash
# Release automation script for autoformalize-math-lab
# Handles version bumping, tagging, and publishing

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
DRY_RUN=${DRY_RUN:-false}
SKIP_TESTS=${SKIP_TESTS:-false}
PYPI_REPO=${PYPI_REPO:-pypi}  # or testpypi
DOCKER_REGISTRY=${DOCKER_REGISTRY:-}
GITHUB_RELEASE=${GITHUB_RELEASE:-true}

show_help() {
    cat << EOF
Usage: $0 [OPTIONS] VERSION_TYPE

Release automation script for autoformalize-math-lab

VERSION_TYPE:
    patch       Bump patch version (x.y.Z)
    minor       Bump minor version (x.Y.0)
    major       Bump major version (X.0.0)
    VERSION     Set specific version (e.g., 1.2.3)

OPTIONS:
    -h, --help          Show this help message
    --dry-run          Show what would be done without making changes
    --skip-tests       Skip running tests before release
    --testpypi         Publish to test PyPI instead of main PyPI
    --no-github        Skip creating GitHub release
    --docker-registry  Docker registry to push to

ENVIRONMENT VARIABLES:
    DRY_RUN            Dry run mode (true|false)
    SKIP_TESTS         Skip tests (true|false)
    PYPI_REPO          PyPI repository (pypi|testpypi)
    DOCKER_REGISTRY    Docker registry URL
    GITHUB_TOKEN       GitHub API token for releases

Examples:
    $0 patch                    # Bump patch version
    $0 minor                    # Bump minor version
    $0 1.2.3                   # Set specific version
    $0 --dry-run patch         # Show what would happen
    $0 --testpypi patch        # Release to test PyPI
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --testpypi)
            PYPI_REPO=testpypi
            shift
            ;;
        --no-github)
            GITHUB_RELEASE=false
            shift
            ;;
        --docker-registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        -*)
            log_error "Unknown option: $1"
            exit 1
            ;;
        *)
            VERSION_TYPE="$1"
            shift
            ;;
    esac
done

if [[ -z "$VERSION_TYPE" ]]; then
    log_error "Version type is required"
    show_help
    exit 1
fi

cd "$PROJECT_ROOT"

# Helper functions
run_command() {
    local cmd="$1"
    local description="$2"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: $cmd"
        return 0
    fi
    
    log_info "$description"
    eval "$cmd"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're on main branch
    local current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [[ "$current_branch" != "main" && "$current_branch" != "master" ]]; then
        log_warning "Current branch is '$current_branch', not main/master"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        log_error "Uncommitted changes detected. Please commit or stash changes first."
        exit 1
    fi
    
    # Check if we can push to remote
    if ! git ls-remote --exit-code origin >/dev/null 2>&1; then
        log_error "Cannot connect to remote repository"
        exit 1
    fi
    
    # Check required tools
    local required_tools=("python3" "pip" "git")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_error "Required tool not found: $tool"
            exit 1
        fi
    done
    
    # Check for bump2version if using semantic versioning
    if [[ "$VERSION_TYPE" =~ ^(patch|minor|major)$ ]]; then
        if ! command -v bump2version >/dev/null 2>&1; then
            log_error "bump2version not found. Install with: pip install bump2version"
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

get_current_version() {
    # Try to get version from pyproject.toml
    if [[ -f "pyproject.toml" ]]; then
        grep -E '^version\s*=' pyproject.toml | sed -E 's/version\s*=\s*"([^"]+)"/\1/'
    elif [[ -f "setup.py" ]]; then
        python3 setup.py --version
    else
        echo "0.0.0"
    fi
}

bump_version() {
    local version_type="$1"
    
    if [[ "$version_type" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        # Specific version provided
        log_info "Setting version to $version_type"
        run_command "bump2version --new-version $version_type patch" "Setting specific version"
    else
        # Semantic version bump
        log_info "Bumping $version_type version"
        run_command "bump2version $version_type" "Bumping version"
    fi
}

run_quality_checks() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_info "Skipping quality checks"
        return 0
    fi
    
    log_info "Running quality checks..."
    
    # Use build script for comprehensive checks
    run_command "$SCRIPT_DIR/build.sh lint" "Running linting checks"
    run_command "$SCRIPT_DIR/build.sh test" "Running test suite"
}

build_distribution() {
    log_info "Building distribution packages..."
    
    run_command "rm -rf build/ dist/" "Cleaning previous builds"
    run_command "python3 -m build" "Building packages"
    
    # Verify packages
    if [[ "$DRY_RUN" != "true" ]]; then
        if [[ ! -d "dist" ]] || [[ -z "$(ls -A dist/)" ]]; then
            log_error "Distribution build failed"
            exit 1
        fi
        log_success "Distribution packages built: $(ls dist/)"
    fi
}

publish_to_pypi() {
    log_info "Publishing to PyPI ($PYPI_REPO)..."
    
    if ! command -v twine >/dev/null 2>&1; then
        log_error "twine not found. Install with: pip install twine"
        exit 1
    fi
    
    local repo_flag=""
    if [[ "$PYPI_REPO" == "testpypi" ]]; then
        repo_flag="--repository testpypi"
    fi
    
    run_command "twine check dist/*" "Checking packages"
    run_command "twine upload $repo_flag dist/*" "Uploading to PyPI"
}

build_and_push_docker() {
    if [[ -z "$DOCKER_REGISTRY" ]]; then
        log_info "No Docker registry specified, skipping Docker build"
        return 0
    fi
    
    log_info "Building and pushing Docker images..."
    
    local version=$(get_current_version)
    local image_name="$DOCKER_REGISTRY/autoformalize-math-lab"
    
    # Build images
    run_command "docker build -t $image_name:$version ." "Building Docker image"
    run_command "docker build -t $image_name:latest ." "Tagging as latest"
    
    # Push images
    run_command "docker push $image_name:$version" "Pushing versioned image"
    run_command "docker push $image_name:latest" "Pushing latest image"
}

create_git_tag() {
    local version=$(get_current_version)
    local tag="v$version"
    
    log_info "Creating Git tag: $tag"
    
    run_command "git add ." "Staging changes"
    run_command "git commit -m \"Release $version\"" "Committing release"
    run_command "git tag -a $tag -m \"Release $version\"" "Creating tag"
    run_command "git push origin main" "Pushing commits"
    run_command "git push origin $tag" "Pushing tag"
}

create_github_release() {
    if [[ "$GITHUB_RELEASE" != "true" ]]; then
        log_info "Skipping GitHub release creation"
        return 0
    fi
    
    if ! command -v gh >/dev/null 2>&1; then
        log_warning "GitHub CLI not found, skipping GitHub release"
        return 0
    fi
    
    local version=$(get_current_version)
    local tag="v$version"
    
    log_info "Creating GitHub release..."
    
    # Generate release notes
    local release_notes="Release $version

## Changes

$(git log --pretty=format:"- %s" $(git describe --tags --abbrev=0 HEAD^)..HEAD)

## Installation

\`\`\`bash
pip install autoformalize-math-lab==$version
\`\`\`

## Docker

\`\`\`bash
docker pull ${DOCKER_REGISTRY:-autoformalize}/autoformalize-math-lab:$version
\`\`\`
"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create GitHub release with notes:"
        echo "$release_notes"
    else
        echo "$release_notes" | gh release create "$tag" --title "Release $version" --notes-file -
    fi
}

post_release_tasks() {
    log_info "Running post-release tasks..."
    
    # Update development dependencies
    run_command "pip install -e .[dev]" "Updating development environment"
    
    # Clean up build artifacts
    run_command "rm -rf build/" "Cleaning build artifacts"
    
    log_success "Post-release tasks completed"
}

# Main release process
main() {
    log_info "Starting release process for version type: $VERSION_TYPE"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "DRY RUN MODE - No changes will be made"
    fi
    
    check_prerequisites
    
    # Show current state
    local current_version=$(get_current_version)
    log_info "Current version: $current_version"
    
    # Run quality checks first
    run_quality_checks
    
    # Bump version
    bump_version "$VERSION_TYPE"
    
    # Show new version
    local new_version=$(get_current_version)
    log_info "New version: $new_version"
    
    # Build distribution
    build_distribution
    
    # Publish to PyPI
    publish_to_pypi
    
    # Build and push Docker images
    build_and_push_docker
    
    # Create Git tag and push
    create_git_tag
    
    # Create GitHub release
    create_github_release
    
    # Post-release tasks
    post_release_tasks
    
    log_success "Release $new_version completed successfully!"
    
    if [[ "$PYPI_REPO" == "testpypi" ]]; then
        log_info "Published to test PyPI. Install with:"
        log_info "pip install -i https://test.pypi.org/simple/ autoformalize-math-lab==$new_version"
    else
        log_info "Published to PyPI. Install with:"
        log_info "pip install autoformalize-math-lab==$new_version"
    fi
}

# Run main function
main "$@"