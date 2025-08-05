"""Unit tests for the core formalization pipeline."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from autoformalize.core.pipeline import FormalizationPipeline, TargetSystem, FormalizationResult
from autoformalize.core.config import FormalizationConfig
from autoformalize.core.exceptions import FormalizationError, UnsupportedSystemError


class TestFormalizationPipeline:
    """Test cases for FormalizationPipeline class."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return FormalizationConfig()

    @pytest.fixture
    def pipeline(self, mock_config):
        """Create a pipeline instance for testing."""
        return FormalizationPipeline(
            target_system=TargetSystem.LEAN4,
            model="gpt-4",
            config=mock_config
        )

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.target_system == TargetSystem.LEAN4
        assert pipeline.model == "gpt-4"
        assert pipeline.config is not None
        assert pipeline.parser is not None

    def test_unsupported_system_raises_error(self):
        """Test that unsupported systems raise an error."""
        with pytest.raises(UnsupportedSystemError):
            FormalizationPipeline(target_system="unsupported_system")

    @pytest.mark.asyncio
    async def test_formalize_empty_content_fails(self, pipeline):
        """Test that empty content fails formalization."""
        with patch.object(pipeline.parser, 'parse') as mock_parse:
            mock_parse.return_value = Mock(theorems=[], definitions=[])
            
            result = await pipeline.formalize("")
            
            assert not result.success
            assert "No mathematical content found" in result.error_message

    @pytest.mark.asyncio
    async def test_formalize_success_path(self, pipeline):
        """Test successful formalization path."""
        # Mock parser
        mock_parsed_content = Mock()
        mock_parsed_content.theorems = [Mock()]
        mock_parsed_content.definitions = []
        
        with patch.object(pipeline.parser, 'parse', return_value=mock_parsed_content) as mock_parse, \
             patch.object(pipeline.generator, 'generate', return_value="theorem test : True := trivial") as mock_generate, \
             patch.object(pipeline.verifier, 'verify', return_value=True) as mock_verify:
            
            latex_content = "\\begin{theorem} Test theorem \\end{theorem}"
            result = await pipeline.formalize(latex_content)
            
            assert result.success
            assert result.formal_code == "theorem test : True := trivial"
            assert result.verification_status is True
            assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_formalize_with_verification_failure(self, pipeline):
        """Test formalization with verification failure."""
        mock_parsed_content = Mock()
        mock_parsed_content.theorems = [Mock()]
        mock_parsed_content.definitions = []
        
        with patch.object(pipeline.parser, 'parse', return_value=mock_parsed_content), \
             patch.object(pipeline.generator, 'generate', return_value="invalid code"), \
             patch.object(pipeline.verifier, 'verify', return_value=False):
            
            latex_content = "\\begin{theorem} Test theorem \\end{theorem}"
            result = await pipeline.formalize(latex_content, verify=True)
            
            assert result.success  # Generation succeeded
            assert result.verification_status is False  # But verification failed

    @pytest.mark.asyncio
    async def test_formalize_file_success(self, pipeline, tmp_path):
        """Test successful file formalization."""
        # Create temporary LaTeX file
        test_file = tmp_path / "test.tex"
        test_file.write_text("\\begin{theorem} Test theorem \\end{theorem}")
        
        output_file = tmp_path / "test.lean"
        
        # Mock the formalize method
        mock_result = FormalizationResult(
            success=True,
            formal_code="theorem test : True := trivial",
            verification_status=True,
            processing_time=1.0
        )
        
        with patch.object(pipeline, 'formalize', return_value=mock_result):
            result = await pipeline.formalize_file(test_file, output_file)
            
            assert result.success
            assert output_file.exists()
            assert output_file.read_text() == "theorem test : True := trivial"

    @pytest.mark.asyncio
    async def test_formalize_file_not_found(self, pipeline):
        """Test formalization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            await pipeline.formalize_file("nonexistent.tex")

    @pytest.mark.asyncio
    async def test_batch_formalize(self, pipeline, tmp_path):
        """Test batch formalization."""
        # Create test files
        files = []
        for i in range(3):
            test_file = tmp_path / f"test{i}.tex"
            test_file.write_text(f"\\begin{{theorem}} Test theorem {i} \\end{{theorem}}")
            files.append(test_file)
        
        output_dir = tmp_path / "output"
        
        # Mock formalize_file to return success
        mock_result = FormalizationResult(success=True, formal_code="theorem test : True := trivial")
        
        with patch.object(pipeline, 'formalize_file', return_value=mock_result):
            results = await pipeline.batch_formalize(files, output_dir, parallel=2)
            
            assert len(results) == 3
            assert all(result.success for result in results)
            assert output_dir.exists()

    def test_get_metrics(self, pipeline):
        """Test metrics retrieval."""
        metrics = pipeline.get_metrics()
        assert isinstance(metrics, dict)

    def test_reset_metrics(self, pipeline):
        """Test metrics reset."""
        pipeline.reset_metrics()
        # Should not raise any exceptions


@pytest.mark.asyncio
class TestFormalizationPipelineIntegration:
    """Integration tests for the formalization pipeline."""

    async def test_full_pipeline_with_mocks(self):
        """Test the full pipeline with mocked dependencies."""
        config = FormalizationConfig()
        pipeline = FormalizationPipeline(target_system="lean4", config=config)
        
        # Mock all dependencies
        mock_parsed_content = Mock()
        mock_parsed_content.theorems = [Mock(name="test_theorem", statement="n + 0 = n")]
        mock_parsed_content.definitions = []
        
        with patch.object(pipeline.parser, 'parse', return_value=mock_parsed_content), \
             patch.object(pipeline.generator, 'generate', return_value="theorem test : ∀ n : ℕ, n + 0 = n := Nat.add_zero"), \
             patch.object(pipeline.verifier, 'verify', return_value=True):
            
            latex_content = """
            \\begin{theorem}
            For any natural number $n$, we have $n + 0 = n$.
            \\end{theorem}
            """
            
            result = await pipeline.formalize(latex_content)
            
            assert result.success
            assert "theorem test" in result.formal_code
            assert result.verification_status is True
            assert result.processing_time > 0
            assert "theorems_count" in result.metrics


class TestTargetSystem:
    """Test cases for TargetSystem enum."""

    def test_target_system_values(self):
        """Test TargetSystem enum values."""
        assert TargetSystem.LEAN4.value == "lean4"
        assert TargetSystem.ISABELLE.value == "isabelle"
        assert TargetSystem.COQ.value == "coq"
        assert TargetSystem.AGDA.value == "agda"

    def test_target_system_from_string(self):
        """Test creating TargetSystem from string."""
        assert TargetSystem("lean4") == TargetSystem.LEAN4
        assert TargetSystem("isabelle") == TargetSystem.ISABELLE


class TestFormalizationResult:
    """Test cases for FormalizationResult dataclass."""

    def test_formalization_result_creation(self):
        """Test FormalizationResult creation."""
        result = FormalizationResult(
            success=True,
            formal_code="theorem test : True := trivial",
            verification_status=True,
            processing_time=1.5
        )
        
        assert result.success
        assert result.formal_code == "theorem test : True := trivial"
        assert result.verification_status is True
        assert result.processing_time == 1.5
        assert result.error_message is None

    def test_formalization_result_failure(self):
        """Test FormalizationResult for failure case."""
        result = FormalizationResult(
            success=False,
            error_message="Generation failed",
            processing_time=0.5
        )
        
        assert not result.success
        assert result.error_message == "Generation failed"
        assert result.formal_code is None
        assert result.verification_status is None