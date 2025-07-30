"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    return TEST_DATA_DIR


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_latex_theorem() -> str:
    """Sample LaTeX theorem for testing."""
    return r"""
\begin{theorem}[Fundamental Theorem of Arithmetic]
Every integer greater than 1 is either prime or can be represented as a unique product of prime numbers.
\end{theorem}
\begin{proof}
The proof follows by strong induction on $n$.
Base case: For $n = 2$, the statement holds as 2 is prime.
Inductive step: Assume the statement holds for all integers $k$ where $2 \leq k < n$.
If $n$ is prime, we are done. Otherwise, $n = ab$ where $1 < a, b < n$.
By the inductive hypothesis, both $a$ and $b$ can be written as products of primes.
Therefore, $n$ is also a product of primes.
\end{proof}
"""


@pytest.fixture
def sample_lean_theorem() -> str:
    """Sample Lean 4 theorem for testing."""
    return """
theorem fundamental_theorem_arithmetic (n : ℕ) (hn : n > 1) :
  ∃ (factors : Multiset ℕ), factors.toList.all Nat.Prime ∧ factors.prod = n :=
by
  -- Implementation would go here
  sorry
"""


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    with patch('openai.OpenAI') as mock:
        client = Mock()
        client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content="Mock LLM response"))
        ]
        mock.return_value = client
        yield client


@pytest.fixture
def mock_proof_assistant():
    """Mock proof assistant for testing."""
    mock_assistant = Mock()
    mock_assistant.verify_proof.return_value = {
        "success": True,
        "errors": [],
        "warnings": []
    }
    return mock_assistant


@pytest.fixture(autouse=True)
def isolate_tests(monkeypatch, temp_dir):
    """Isolate tests by setting temporary directories."""
    monkeypatch.setenv("AUTOFORMALIZE_CACHE_DIR", str(temp_dir / "cache"))
    monkeypatch.setenv("AUTOFORMALIZE_OUTPUT_DIR", str(temp_dir / "output"))
    monkeypatch.setenv("AUTOFORMALIZE_LOG_DIR", str(temp_dir / "logs"))


# Pytest markers for different test categories
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (deselect with '-m \"not unit\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "mathematical: marks tests that verify mathematical correctness"
    )
    config.addinivalue_line(
        "markers", "lean: marks tests specific to Lean 4"
    )
    config.addinivalue_line(
        "markers", "isabelle: marks tests specific to Isabelle/HOL"
    )
    config.addinivalue_line(
        "markers", "coq: marks tests specific to Coq"
    )
    config.addinivalue_line(
        "markers", "llm: marks tests that require LLM API access"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "expensive: marks tests that consume significant resources"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    for item in items:
        # Add unit marker to tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add e2e marker to tests in e2e/ directory
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add slow marker to tests that might be slow
        if any(keyword in item.nodeid.lower() for keyword in ["large", "performance", "benchmark"]):
            item.add_marker(pytest.mark.slow)