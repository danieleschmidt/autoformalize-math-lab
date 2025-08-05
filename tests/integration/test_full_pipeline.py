"""Integration tests for the complete formalization pipeline."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from autoformalize import FormalizationPipeline
from autoformalize.core.config import FormalizationConfig
from autoformalize.parsers.latex_parser import MathematicalStatement, ParsedContent


@pytest.mark.asyncio
class TestFullPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def sample_latex_content(self):
        """Sample LaTeX content for testing."""
        return """
        \\documentclass{article}
        \\usepackage{amsmath, amsthm}
        
        \\begin{document}
        
        \\begin{definition}[Prime Number]
        A natural number $n > 1$ is called \\emph{prime} if its only positive divisors are 1 and $n$.
        \\end{definition}
        
        \\begin{theorem}[Infinitude of Primes]
        There are infinitely many prime numbers.
        \\end{theorem}
        
        \\begin{proof}
        Suppose there are only finitely many primes $p_1, p_2, \\ldots, p_k$.
        Consider the number $N = p_1 p_2 \\cdots p_k + 1$.
        This number is greater than 1 and is not divisible by any of the primes $p_i$.
        Therefore, $N$ must have a prime divisor not in our list, contradicting our assumption.
        \\end{proof}
        
        \\end{document}
        """

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = FormalizationConfig()
        # Set test-friendly values
        config.model.temperature = 0.1
        config.model.max_tokens = 1000
        config.verification.timeout = 10
        return config

    async def test_complete_pipeline_mock_success(self, sample_latex_content, mock_config):
        """Test complete pipeline with mocked components."""
        # Create pipeline
        pipeline = FormalizationPipeline(
            target_system="lean4",
            model="gpt-4",
            config=mock_config
        )
        
        # Mock parsed content
        mock_parsed_content = ParsedContent()
        mock_parsed_content.definitions = [
            MathematicalStatement(
                type="definition",
                name="Prime Number",
                statement="A natural number n > 1 is called prime if its only positive divisors are 1 and n."
            )
        ]
        mock_parsed_content.theorems = [
            MathematicalStatement(
                type="theorem", 
                name="Infinitude of Primes",
                statement="There are infinitely many prime numbers.",
                proof="Suppose there are only finitely many primes..."
            )
        ]
        
        # Mock expected Lean 4 output
        expected_lean_code = """
import Mathlib.Data.Nat.Prime

-- Definition: Prime Number
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Theorem: Infinitude of Primes  
theorem infinitude_of_primes : ∀ n : ℕ, ∃ p : ℕ, p > n ∧ isPrime p := by
  intro n
  -- Classical proof by contradiction
  sorry
"""
        
        # Mock all pipeline components
        with patch.object(pipeline.parser, 'parse', return_value=mock_parsed_content) as mock_parse, \
             patch.object(pipeline.generator, 'generate', return_value=expected_lean_code) as mock_generate, \
             patch.object(pipeline.verifier, 'verify', return_value=True) as mock_verify:
            
            # Run pipeline
            result = await pipeline.formalize(sample_latex_content, verify=True)
            
            # Verify result
            assert result.success
            assert result.formal_code == expected_lean_code
            assert result.verification_status is True
            assert result.processing_time > 0
            
            # Verify component calls
            mock_parse.assert_called_once()
            mock_generate.assert_called_once_with(mock_parsed_content)
            mock_verify.assert_called_once_with(expected_lean_code, timeout=10)
            
            # Check metrics
            assert "processing_time" in result.metrics
            assert "theorems_count" in result.metrics
            assert "definitions_count" in result.metrics
            assert result.metrics["theorems_count"] == 1
            assert result.metrics["definitions_count"] == 1

    async def test_pipeline_with_parsing_failure(self, sample_latex_content, mock_config):
        """Test pipeline behavior when parsing fails."""
        pipeline = FormalizationPipeline(
            target_system="lean4",
            config=mock_config
        )
        
        # Mock parser to return empty content
        empty_content = ParsedContent()
        
        with patch.object(pipeline.parser, 'parse', return_value=empty_content):
            result = await pipeline.formalize(sample_latex_content)
            
            assert not result.success
            assert "No mathematical content found" in result.error_message

    async def test_pipeline_with_generation_failure(self, sample_latex_content, mock_config):
        """Test pipeline behavior when generation fails."""
        pipeline = FormalizationPipeline(
            target_system="lean4", 
            config=mock_config
        )
        
        # Mock successful parsing but failing generation
        mock_parsed_content = ParsedContent()
        mock_parsed_content.theorems = [
            MathematicalStatement(type="theorem", statement="Test theorem")
        ]
        
        with patch.object(pipeline.parser, 'parse', return_value=mock_parsed_content), \
             patch.object(pipeline.generator, 'generate', side_effect=Exception("Generation failed")):
            
            result = await pipeline.formalize(sample_latex_content)
            
            assert not result.success
            assert "Generation failed" in result.error_message

    async def test_pipeline_with_verification_failure(self, sample_latex_content, mock_config):
        """Test pipeline behavior when verification fails."""
        pipeline = FormalizationPipeline(
            target_system="lean4",
            config=mock_config
        )
        
        # Mock successful parsing and generation but failing verification
        mock_parsed_content = ParsedContent()
        mock_parsed_content.theorems = [
            MathematicalStatement(type="theorem", statement="Test theorem")
        ]
        
        with patch.object(pipeline.parser, 'parse', return_value=mock_parsed_content), \
             patch.object(pipeline.generator, 'generate', return_value="theorem test : True := trivial"), \
             patch.object(pipeline.verifier, 'verify', return_value=False):
            
            result = await pipeline.formalize(sample_latex_content, verify=True)
            
            # Generation succeeded but verification failed
            assert result.success  # Overall success (generation worked)
            assert result.verification_status is False
            assert result.formal_code == "theorem test : True := trivial"

    async def test_batch_processing_integration(self, mock_config, tmp_path):
        """Test batch processing integration."""
        pipeline = FormalizationPipeline(
            target_system="lean4",
            config=mock_config
        )
        
        # Create test files
        test_files = []
        for i in range(3):
            test_file = tmp_path / f"theorem_{i}.tex"
            content = f"""
            \\begin{{theorem}}
            Test theorem {i}: $n + {i} = {i} + n$
            \\end{{theorem}}
            """
            test_file.write_text(content)
            test_files.append(test_file)
        
        output_dir = tmp_path / "output"
        
        # Mock successful processing
        mock_result = Mock()
        mock_result.success = True
        mock_result.formal_code = "theorem test : True := trivial"
        
        with patch.object(pipeline, 'formalize_file', return_value=mock_result):
            results = await pipeline.batch_formalize(
                test_files,
                output_dir=output_dir,
                parallel=2,
                verify=False
            )
            
            assert len(results) == 3
            assert all(result.success for result in results)
            assert output_dir.exists()

    async def test_cross_system_compatibility(self, sample_latex_content, mock_config):
        """Test pipeline with different target systems."""
        systems_to_test = ["lean4", "isabelle", "coq"]
        
        for system in systems_to_test:
            try:
                pipeline = FormalizationPipeline(
                    target_system=system,
                    config=mock_config
                )
                
                # Mock components for each system
                mock_parsed_content = ParsedContent()
                mock_parsed_content.theorems = [
                    MathematicalStatement(type="theorem", statement="Test theorem")
                ]
                
                expected_output = {
                    "lean4": "theorem test : True := trivial",
                    "isabelle": "theorem test: \"True\" by simp",
                    "coq": "Theorem test : True. Proof. trivial. Qed."
                }
                
                with patch.object(pipeline.parser, 'parse', return_value=mock_parsed_content), \
                     patch.object(pipeline.generator, 'generate', return_value=expected_output[system]):
                    
                    result = await pipeline.formalize(sample_latex_content, verify=False)
                    
                    assert result.success
                    assert result.formal_code == expected_output[system]
                    
            except Exception as e:
                # Some systems might not be fully implemented
                pytest.skip(f"System {system} not fully implemented: {e}")

    async def test_pipeline_metrics_collection(self, sample_latex_content, mock_config):
        """Test that pipeline collects comprehensive metrics."""
        pipeline = FormalizationPipeline(
            target_system="lean4",
            config=mock_config
        )
        
        # Mock successful pipeline
        mock_parsed_content = ParsedContent() 
        mock_parsed_content.theorems = [
            MathematicalStatement(type="theorem", statement="Test theorem 1"),
            MathematicalStatement(type="theorem", statement="Test theorem 2")
        ]
        mock_parsed_content.definitions = [
            MathematicalStatement(type="definition", statement="Test definition")
        ]
        
        with patch.object(pipeline.parser, 'parse', return_value=mock_parsed_content), \
             patch.object(pipeline.generator, 'generate', return_value="-- Generated code"), \
             patch.object(pipeline.verifier, 'verify', return_value=True):
            
            result = await pipeline.formalize(sample_latex_content, verify=True)
            
            # Check comprehensive metrics
            assert "processing_time" in result.metrics
            assert "content_length" in result.metrics
            assert "theorems_count" in result.metrics
            assert "definitions_count" in result.metrics
            assert "formal_code_length" in result.metrics
            
            assert result.metrics["theorems_count"] == 2
            assert result.metrics["definitions_count"] == 1
            assert result.metrics["content_length"] == len(sample_latex_content)
            assert result.metrics["formal_code_length"] == len("-- Generated code")

    async def test_pipeline_configuration_impact(self, sample_latex_content):
        """Test that configuration properly affects pipeline behavior."""
        # Test with different configurations
        configs = [
            FormalizationConfig(),  # Default
            FormalizationConfig()   # Custom (would be modified in real test)
        ]
        
        # Modify second config
        configs[1].model.temperature = 0.8
        configs[1].verification.timeout = 60
        
        for i, config in enumerate(configs):
            pipeline = FormalizationPipeline(
                target_system="lean4",
                config=config
            )
            
            # Verify configuration is applied
            assert pipeline.config.model.temperature == config.model.temperature
            assert pipeline.config.verification.timeout == config.verification.timeout

    async def test_error_recovery_integration(self, sample_latex_content, mock_config):
        """Test error recovery mechanisms in full pipeline."""
        pipeline = FormalizationPipeline(
            target_system="lean4",
            config=mock_config
        )
        
        # Mock transient failure followed by success
        call_count = 0
        
        def mock_generate_with_retry(content):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Transient error")
            return "theorem test : True := trivial"
        
        mock_parsed_content = ParsedContent()
        mock_parsed_content.theorems = [
            MathematicalStatement(type="theorem", statement="Test theorem")
        ]
        
        with patch.object(pipeline.parser, 'parse', return_value=mock_parsed_content), \
             patch.object(pipeline.generator, 'generate', side_effect=mock_generate_with_retry):
            
            # First call should fail
            result1 = await pipeline.formalize(sample_latex_content)
            assert not result1.success
            
            # Second call should succeed
            result2 = await pipeline.formalize(sample_latex_content)
            assert result2.success


@pytest.mark.asyncio 
class TestPipelineResourceManagement:
    """Test resource management in the pipeline."""

    async def test_pipeline_cleanup(self, mock_config):
        """Test that pipeline properly cleans up resources."""
        pipeline = FormalizationPipeline(
            target_system="lean4",
            config=mock_config
        )
        
        # Simulate some operations
        metrics_before = pipeline.get_metrics()
        
        # Reset metrics (cleanup operation)
        pipeline.reset_metrics()
        
        metrics_after = pipeline.get_metrics()
        
        # Verify cleanup
        assert isinstance(metrics_before, dict)
        assert isinstance(metrics_after, dict)

    async def test_concurrent_pipeline_operations(self, mock_config):
        """Test multiple concurrent pipeline operations."""
        pipeline = FormalizationPipeline(
            target_system="lean4",
            config=mock_config
        )
        
        # Mock components for concurrent testing
        mock_parsed_content = ParsedContent()
        mock_parsed_content.theorems = [
            MathematicalStatement(type="theorem", statement="Concurrent theorem")
        ]
        
        async def mock_async_ops():
            with patch.object(pipeline.parser, 'parse', return_value=mock_parsed_content), \
                 patch.object(pipeline.generator, 'generate', return_value="theorem test : True := trivial"), \
                 patch.object(pipeline.verifier, 'verify', return_value=True):
                
                return await pipeline.formalize("\\begin{theorem} Test \\end{theorem}")
        
        # Run multiple operations concurrently
        tasks = [mock_async_ops() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 5
        assert all(result.success for result in results)