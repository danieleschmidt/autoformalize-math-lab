# Testing Guide

This directory contains comprehensive tests for the Autoformalize Math Lab project.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── fixtures/                # Test data and sample theorems
│   ├── __init__.py
│   └── sample_theorems.py   # Mathematical theorems for testing
├── unit/                    # Unit tests for individual components
│   ├── test_parsers.py      # LaTeX, PDF, arXiv parsers
│   ├── test_generators.py   # Formal proof generators
│   └── test_cli.py          # Command-line interface
├── integration/             # Integration tests
│   ├── test_formalization_pipeline.py
│   └── test_end_to_end_pipeline.py
├── e2e/                     # End-to-end tests
│   └── test_cli_workflows.py
├── performance/             # Performance and benchmark tests
│   └── test_benchmarks.py
├── security/                # Security tests
│   └── test_security.py
└── data/                    # Test data files
    └── README.md
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/autoformalize --cov-report=html

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Exclude slow tests
pytest -m "mathematical"    # Mathematical correctness tests
```

### Test Categories

Tests are organized using pytest markers:

- **`unit`**: Fast, isolated tests of individual components
- **`integration`**: Tests of component interactions
- **`e2e`**: Complete workflow tests
- **`mathematical`**: Tests verifying mathematical correctness
- **`performance`**: Performance benchmarks and load tests
- **`slow`**: Time-consuming tests (excluded by default in CI)
- **`expensive`**: Resource-intensive tests
- **`security`**: Security and vulnerability tests
- **`llm`**: Tests requiring LLM API access
- **`lean/isabelle/coq`**: Proof assistant specific tests

### Environment-Specific Testing

```bash
# Development testing (fast feedback)
pytest -m "unit and not slow" --maxfail=1

# CI testing (comprehensive but time-limited)
pytest -m "not expensive and not llm" --durations=10

# Full test suite (all tests including slow ones)
pytest -m "not llm" --tb=short

# Performance testing
pytest -m performance --benchmark-only

# Security testing
pytest -m security --tb=line
```

## Test Configuration

### Environment Variables

Set these environment variables for testing:

```bash
export AUTOFORMALIZE_ENV=testing
export AUTOFORMALIZE_LOG_LEVEL=DEBUG
export TEST_DATABASE_URL=postgresql://test_user:test_pass@localhost/test_db
export TEST_REDIS_URL=redis://localhost:6379/1

# For LLM tests (optional)
export OPENAI_API_KEY=your_test_key
export ANTHROPIC_API_KEY=your_test_key
```

### Test Data

Test data is organized in `tests/fixtures/`:

- **`sample_theorems.py`**: Mathematical theorems in LaTeX and formal languages
- **`error_samples.py`**: Common error cases for testing error handling
- **`benchmark_data.py`**: Data for performance testing

### Mocking Strategy

Tests use extensive mocking to avoid external dependencies:

- **LLM APIs**: Mocked to return predictable responses
- **Proof Assistants**: Mocked to simulate verification results
- **File System**: Temporary directories for isolation
- **Network**: Mocked HTTP requests for arXiv integration

## Writing New Tests

### Test File Structure

```python
"""Test module docstring."""

import pytest
from unittest.mock import Mock, patch

from autoformalize.component import ComponentToTest
from tests.fixtures import SAMPLE_DATA


@pytest.mark.unit
class TestComponentName:
    """Test cases for ComponentToTest."""
    
    def test_basic_functionality(self, fixture_name):
        """Test description."""
        # Arrange
        component = ComponentToTest()
        
        # Act
        result = component.method(input_data)
        
        # Assert
        assert result is not None
        assert result.property == expected_value
    
    @pytest.mark.slow
    def test_performance_aspect(self):
        """Test that may take longer to execute."""
        pass
    
    @pytest.mark.mathematical
    def test_mathematical_correctness(self):
        """Test that verifies mathematical accuracy."""
        pass
```

### Fixture Usage

Common fixtures are defined in `conftest.py`:

```python
def test_with_fixtures(temp_dir, sample_latex_theorem, mock_llm_client):
    """Example using multiple fixtures."""
    # temp_dir: temporary directory for file operations
    # sample_latex_theorem: sample LaTeX content
    # mock_llm_client: mocked LLM client
    pass
```

### Test Naming

Follow these conventions:

- `test_[functionality]` for basic tests
- `test_[functionality]_with_[condition]` for specific scenarios
- `test_[functionality]_error_handling` for error cases
- `test_[functionality]_edge_cases` for boundary conditions

## Test Data Management

### Adding New Test Theorems

Add mathematical content to `tests/fixtures/sample_theorems.py`:

```python
NEW_THEOREM = {
    "theorem_name": {
        "latex": r"LaTeX source...",
        "lean4": "Lean 4 formalization...",
        "isabelle": "Isabelle formalization...",
        "domain": "mathematical_domain",
        "difficulty": "undergraduate|graduate|research"
    }
}
```

### Creating Test Fixtures

For complex test data, create dedicated fixture files:

```python
# tests/fixtures/my_test_data.py
@pytest.fixture
def my_complex_fixture():
    """Fixture description."""
    return complex_test_data
```

## Continuous Integration

### Pre-commit Hooks

Tests run automatically via pre-commit hooks:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### GitHub Actions

The CI pipeline runs different test suites:

- **Pull Requests**: Fast unit and integration tests
- **Main Branch**: Full test suite including performance tests
- **Nightly**: Complete test suite including expensive tests

### Test Coverage

Maintain high test coverage:

- **Target**: 85%+ overall coverage
- **Critical Components**: 95%+ coverage required
- **New Code**: Must not decrease overall coverage

## Debugging Tests

### Running Individual Tests

```bash
# Run specific test
pytest tests/unit/test_parsers.py::TestLaTeXParser::test_parse_simple_theorem

# Run with debugging
pytest --pdb tests/unit/test_parsers.py::test_failing_test

# Run with verbose output
pytest -vvv --tb=long tests/unit/test_parsers.py
```

### Test Debugging Tips

1. **Use `pytest.set_trace()`** for interactive debugging
2. **Check fixture values** with `print()` statements
3. **Verify mock calls** with `mock_object.assert_called_with()`
4. **Isolate failures** by running single tests

### Common Issues

- **Import Errors**: Check `PYTHONPATH` includes `src/`
- **Fixture Conflicts**: Ensure fixture scopes are appropriate
- **Mock Issues**: Verify mock patches target the correct module
- **Async Tests**: Use `pytest-asyncio` for async test functions

## Performance Testing

### Benchmark Tests

Performance tests measure:

- **Latency**: Single operation execution time
- **Throughput**: Operations per second
- **Memory Usage**: RAM consumption patterns
- **CPU Utilization**: Processor usage efficiency

### Running Benchmarks

```bash
# Run performance tests
pytest -m performance

# Generate performance report
pytest -m performance --benchmark-json=benchmark.json

# Compare with baseline
pytest -m performance --benchmark-compare=baseline.json
```

## Security Testing

### Security Test Coverage

- Input validation and sanitization
- API key and secret handling
- File system security
- Network request validation
- Output sanitization

### Running Security Tests

```bash
# Run security tests
pytest -m security

# Run with security scanner
bandit -r src/ tests/

# Check for vulnerabilities
safety check
```

## Contributing Test Improvements

When contributing tests:

1. **Follow naming conventions**
2. **Add appropriate markers**
3. **Include docstrings**
4. **Mock external dependencies**
5. **Test both success and failure cases**
6. **Update this documentation**

For questions about testing, see the [Contributing Guide](../CONTRIBUTING.md) or open an issue on GitHub.