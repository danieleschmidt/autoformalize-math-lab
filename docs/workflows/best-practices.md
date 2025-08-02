# GitHub Actions Best Practices

## Overview

This document outlines best practices for implementing and maintaining GitHub Actions workflows in the autoformalize-math-lab project, with specific considerations for mathematical formalization and LLM-based systems.

## Workflow Design Principles

### 1. Separation of Concerns

**✅ Good Practice:**
```yaml
jobs:
  lint:
    name: Code Quality
    # Only linting and formatting
  
  test:
    name: Unit Tests
    # Only unit testing
    
  security:
    name: Security Scan
    # Only security-related checks
```

**❌ Avoid:**
```yaml
jobs:
  everything:
    name: Do Everything
    steps:
      - name: Lint, test, build, deploy, and make coffee
```

### 2. Fail-Fast Strategy

Configure jobs to fail quickly when issues are detected:

```yaml
strategy:
  fail-fast: true  # Stop other jobs when one fails
  matrix:
    python-version: ["3.9", "3.10", "3.11", "3.12"]
```

### 3. Conditional Execution

Use path-based and context-based conditions to optimize workflow execution:

```yaml
# Only run mathematical tests when relevant files change
mathematical-tests:
  if: |
    contains(github.event.pull_request.changed_files, '.tex') ||
    contains(github.event.pull_request.changed_files, '.lean') ||
    contains(github.event.pull_request.changed_files, 'src/autoformalize/generators/') ||
    github.event_name == 'push'
```

## Performance Optimization

### 1. Effective Caching

**Python Dependencies:**
```yaml
- name: Cache pip dependencies
  uses: actions/cache@v3
  with:
    path: |
      ~/.cache/pip
      ~/Library/Caches/pip  # macOS
      ~\AppData\Local\pip\Cache  # Windows
    key: ${{ runner.os }}-pip-${{ matrix.python-version }}-${{ hashFiles('**/requirements.txt', '**/pyproject.toml') }}
    restore-keys: |
      ${{ runner.os }}-pip-${{ matrix.python-version }}-
      ${{ runner.os }}-pip-
```

**Proof Assistant Installations:**
```yaml
- name: Cache Lean installation
  uses: actions/cache@v3
  with:
    path: ~/.elan
    key: lean-${{ runner.os }}-${{ hashFiles('lean-toolchain') }}
    restore-keys: lean-${{ runner.os }}-
```

**Build Artifacts:**
```yaml
- name: Cache build artifacts
  uses: actions/cache@v3
  with:
    path: |
      dist/
      build/
      *.egg-info/
    key: build-${{ runner.os }}-${{ github.sha }}
    restore-keys: build-${{ runner.os }}-
```

### 2. Matrix Optimization

**Reduce Matrix Size for Expensive Tests:**
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python-version: ["3.9", "3.10", "3.11", "3.12"]
    exclude:
      # Reduce combinations for non-critical OS/version pairs
      - os: macos-latest
        python-version: "3.9"
      - os: windows-latest
        python-version: "3.9"
```

**Targeted Matrix for Different Test Types:**
```yaml
# Fast tests on all combinations
unit-tests:
  strategy:
    matrix:
      os: [ubuntu-latest, macos-latest, windows-latest]
      python-version: ["3.9", "3.10", "3.11", "3.12"]

# Expensive tests only on primary platform
integration-tests:
  strategy:
    matrix:
      os: [ubuntu-latest]
      python-version: ["3.11"]
```

### 3. Parallel Job Execution

```yaml
jobs:
  test:
    name: Test Suite
    runs-on: ubuntu-latest
    steps:
      - name: Run tests in parallel
        run: |
          pytest tests/ -n auto --dist=worksteal
```

## Security Best Practices

### 1. Secrets Management

**✅ Proper Secret Usage:**
```yaml
- name: Use API keys securely
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_TEST_API_KEY }}
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_TEST_API_KEY }}
  run: |
    python test_llm_integration.py
```

**❌ Avoid:**
```yaml
- name: Dangerous secret exposure
  run: |
    echo "API Key: ${{ secrets.OPENAI_API_KEY }}"  # DON'T DO THIS
    curl -H "Authorization: Bearer ${{ secrets.OPENAI_API_KEY }}" ...  # Visible in logs
```

### 2. Minimal Permissions

```yaml
permissions:
  contents: read      # Only read repository contents
  checks: write       # Write check results
  pull-requests: write # Comment on PRs
  # Don't grant unnecessary permissions
```

### 3. Dependency Verification

```yaml
- name: Verify dependencies
  run: |
    pip install pip-audit safety
    pip-audit --format=json --output=audit-report.json
    safety check --json --output=safety-report.json
```

## Mathematical Formalization Specific Practices

### 1. Proof Assistant Integration

**Robust Proof Verification:**
```yaml
- name: Install Lean 4 with retries
  run: |
    for i in {1..3}; do
      curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y && break
      echo "Retry $i failed, waiting..."
      sleep 10
    done
    echo "$HOME/.elan/bin" >> $GITHUB_PATH

- name: Verify Lean installation
  run: |
    lean --version
    lake --version
```

**Timeout Management for Mathematical Tests:**
```yaml
- name: Run mathematical verification
  timeout-minutes: 30  # Prevent hanging on complex proofs
  run: |
    pytest tests/mathematical/ -v --timeout=1800  # 30 minutes per test
```

### 2. LLM API Integration

**Rate Limiting and Retry Logic:**
```yaml
- name: Test LLM integration with retries
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_TEST_API_KEY }}
  run: |
    python -c "
    import time
    import openai
    from tenacity import retry, stop_after_attempt, wait_exponential
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def test_api():
        # Your LLM test code here
        pass
    
    test_api()
    "
```

**Cost Management:**
```yaml
- name: Monitor API costs
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_TEST_API_KEY }}
  run: |
    # Use lightweight models for testing
    python tests/test_llm_basic.py --model=gpt-3.5-turbo
    
    # Track token usage
    python scripts/track_api_usage.py
```

## Error Handling and Debugging

### 1. Comprehensive Error Information

```yaml
- name: Run tests with detailed output
  run: |
    pytest tests/ -v --tb=long --show-capture=all
  continue-on-error: false
  
- name: Upload test results on failure
  if: failure()
  uses: actions/upload-artifact@v3
  with:
    name: test-results-${{ matrix.python-version }}
    path: |
      pytest-report.xml
      coverage.xml
      logs/
```

### 2. Debug Information Collection

```yaml
- name: Collect debug information
  if: failure()
  run: |
    echo "System information:"
    uname -a
    python --version
    pip list
    
    echo "Environment variables:"
    env | grep -E "(PYTHON|PATH|CI)" | sort
    
    echo "Process information:"
    ps aux | grep python
    
    echo "Disk usage:"
    df -h
```

### 3. Incremental Debugging

```yaml
- name: Test individual components
  run: |
    # Test each component separately for easier debugging
    pytest tests/unit/test_parsers.py -v
    pytest tests/unit/test_generators.py -v
    pytest tests/unit/test_verifiers.py -v
```

## Monitoring and Maintenance

### 1. Workflow Health Monitoring

```yaml
- name: Report workflow metrics
  if: always()
  run: |
    echo "Job duration: ${{ github.event.workflow_run.run_started_at }} to $(date)"
    echo "Runner: ${{ runner.os }} ${{ runner.arch }}"
    echo "Status: ${{ job.status }}"
```

### 2. Dependency Updates

**Automated Dependency Updates:**
```yaml
# .github/workflows/dependency-update.yml
name: Update Dependencies

on:
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Update requirements
        run: |
          pip install pip-tools
          pip-compile --upgrade requirements.in
          pip-compile --upgrade requirements-dev.in
```

### 3. Performance Monitoring

```yaml
- name: Track workflow performance
  run: |
    START_TIME=${{ github.event.workflow_run.run_started_at }}
    END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Calculate duration and log metrics
    python scripts/log_workflow_metrics.py \
      --start="$START_TIME" \
      --end="$END_TIME" \
      --workflow="${{ github.workflow }}" \
      --job="${{ github.job }}"
```

## Documentation and Communication

### 1. Meaningful Job and Step Names

**✅ Good:**
```yaml
jobs:
  unit-tests:
    name: Unit Tests (Python ${{ matrix.python-version }})
    steps:
      - name: Install dependencies for Python ${{ matrix.python-version }}
      - name: Run unit tests with coverage reporting
```

**❌ Avoid:**
```yaml
jobs:
  job1:
    name: Tests
    steps:
      - name: Install stuff
      - name: Run tests
```

### 2. Status Communication

```yaml
- name: Comment PR with test results
  if: github.event_name == 'pull_request'
  uses: actions/github-script@v6
  with:
    script: |
      const { execSync } = require('child_process');
      
      // Generate test summary
      const testResults = execSync('python scripts/generate_test_summary.py').toString();
      
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: `## Test Results\n\n${testResults}`
      });
```

### 3. Badge Integration

Add workflow status badges to README.md:

```markdown
[![CI](https://github.com/username/autoformalize-math-lab/workflows/CI/badge.svg)](https://github.com/username/autoformalize-math-lab/actions)
[![Security](https://github.com/username/autoformalize-math-lab/workflows/Security/badge.svg)](https://github.com/username/autoformalize-math-lab/actions)
[![Performance](https://github.com/username/autoformalize-math-lab/workflows/Performance/badge.svg)](https://github.com/username/autoformalize-math-lab/actions)
```

## Common Anti-Patterns to Avoid

### 1. Overly Complex Workflows

**❌ Don't:**
```yaml
# Single workflow that does everything
name: Everything
on: [push, pull_request, schedule, workflow_dispatch, release, ...]
jobs:
  mega-job:
    # 500 lines of steps
```

**✅ Do:**
```yaml
# Separate workflows for different purposes
name: CI              # .github/workflows/ci.yml
name: Security        # .github/workflows/security.yml
name: Release         # .github/workflows/release.yml
name: Performance     # .github/workflows/performance.yml
```

### 2. Hardcoded Values

**❌ Don't:**
```yaml
- name: Install Python 3.11
  uses: actions/setup-python@v4
  with:
    python-version: "3.11"  # Hardcoded everywhere
```

**✅ Do:**
```yaml
env:
  PYTHON_DEFAULT_VERSION: "3.11"  # Defined once, used everywhere

- name: Install Python ${{ env.PYTHON_DEFAULT_VERSION }}
  uses: actions/setup-python@v4
  with:
    python-version: ${{ env.PYTHON_DEFAULT_VERSION }}
```

### 3. Ignoring Failures

**❌ Don't:**
```yaml
- name: Run tests
  run: pytest tests/ || true  # Ignores test failures
```

**✅ Do:**
```yaml
- name: Run tests
  run: pytest tests/
  continue-on-error: false  # Explicit failure handling

- name: Upload test results even on failure
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test-results.xml
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **Workflow Timeouts**
   - Increase timeout values for mathematical verification
   - Use parallel execution where possible
   - Cache dependencies and build artifacts

2. **Flaky Tests**
   - Implement retry logic for LLM API calls
   - Use test isolation and cleanup
   - Add debugging output for intermittent failures

3. **Resource Limitations**
   - Monitor memory usage in mathematical computations
   - Use matrix exclusions to reduce resource consumption
   - Implement test sampling for expensive operations

4. **API Rate Limiting**
   - Use test API keys with appropriate limits
   - Implement exponential backoff
   - Cache API responses where possible

This guide provides a foundation for implementing robust, maintainable, and efficient GitHub Actions workflows specifically tailored for mathematical formalization projects.