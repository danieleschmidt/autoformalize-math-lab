"""Integration tests for the complete formalization pipeline."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import json

from autoformalize.core.pipeline import FormalizationPipeline
from autoformalize.core.correction import SelfCorrectingPipeline
from tests.fixtures import ALL_THEOREMS


@pytest.mark.integration
class TestFormalizationPipeline:
    """Test the complete formalization pipeline."""
    
    @pytest.fixture
    def pipeline(self, mock_llm_client, mock_proof_assistant):
        """Create a pipeline instance for testing."""
        return FormalizationPipeline(
            target_system="lean4",
            llm_client=mock_llm_client,
            proof_assistant=mock_proof_assistant,
            use_cache=False  # Disable caching for tests
        )
    
    def test_latex_to_lean_pipeline(self, pipeline):
        """Test complete LaTeX to Lean formalization."""
        theorem_data = ALL_THEOREMS["quadratic_formula"]
        
        # Mock successful pipeline execution
        pipeline.llm_client.chat.completions.create.return_value.choices[0].message.content = \
            theorem_data["lean4"]
        pipeline.proof_assistant.verify_proof.return_value = {
            "success": True,
            "errors": [],
            "warnings": []
        }
        
        result = pipeline.formalize(theorem_data["latex"])
        
        assert result is not None
        assert result["success"] == True
        assert result["formal_proof"] is not None
        assert "theorem" in result["formal_proof"]
    
    def test_pipeline_with_self_correction(self, mock_llm_client, mock_proof_assistant):
        """Test pipeline with self-correction enabled."""
        pipeline = SelfCorrectingPipeline(
            target_system="lean4",
            llm_client=mock_llm_client,
            proof_assistant=mock_proof_assistant,
            max_correction_rounds=3
        )
        
        theorem_data = ALL_THEOREMS["binomial_theorem"]
        
        # Mock initial failure, then success
        mock_proof_assistant.verify_proof.side_effect = [
            {"success": False, "errors": ["syntax error"], "warnings": []},
            {"success": True, "errors": [], "warnings": []}
        ]
        
        # Mock LLM responses for initial attempt and correction
        mock_llm_client.chat.completions.create.side_effect = [
            Mock(choices=[Mock(message=Mock(content="theorem initial : wrong syntax"))]),
            Mock(choices=[Mock(message=Mock(content=theorem_data["lean4"]))])
        ]
        
        result = pipeline.formalize_with_feedback(theorem_data["latex"])
        
        assert result["success"] == True
        assert result["correction_rounds"] == 1
        assert result["final_proof"] is not None
    
    def test_pipeline_failure_handling(self, pipeline):
        """Test pipeline handles various failure modes."""
        # Test parsing failure
        invalid_latex = "\\invalid{latex"
        with pytest.raises(Exception):
            pipeline.formalize(invalid_latex)
        
        # Test LLM failure
        pipeline.llm_client.chat.completions.create.side_effect = Exception("API Error")
        with pytest.raises(Exception):
            pipeline.formalize("\\begin{theorem}Test\\end{theorem}")
    
    def test_pipeline_with_different_targets(self, mock_llm_client, mock_proof_assistant):
        """Test pipeline with different target proof assistants."""
        targets = ["lean4", "isabelle", "coq"]
        theorem_data = ALL_THEOREMS["euclid_infinitude_primes"]
        
        for target in targets:
            pipeline = FormalizationPipeline(
                target_system=target,
                llm_client=mock_llm_client,
                proof_assistant=mock_proof_assistant
            )
            
            mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
                f"// {target} proof would go here"
            mock_proof_assistant.verify_proof.return_value = {
                "success": True,
                "errors": [],
                "warnings": []
            }
            
            result = pipeline.formalize(theorem_data["latex"])
            
            assert result["success"] == True
            assert target in result["target_system"]
    
    def test_batch_processing(self, pipeline):
        """Test batch processing of multiple theorems."""
        theorems = [
            ALL_THEOREMS["quadratic_formula"]["latex"],
            ALL_THEOREMS["binomial_theorem"]["latex"]
        ]
        
        # Mock successful processing for all theorems
        pipeline.llm_client.chat.completions.create.return_value.choices[0].message.content = \
            "theorem batch_result : True := trivial"
        pipeline.proof_assistant.verify_proof.return_value = {
            "success": True,
            "errors": [],
            "warnings": []
        }
        
        results = pipeline.formalize_batch(theorems)
        
        assert len(results) == len(theorems)
        for result in results:
            assert result["success"] == True
    
    def test_pipeline_with_context(self, pipeline):
        """Test pipeline with additional mathematical context."""
        theorem_data = ALL_THEOREMS["fundamental_theorem_arithmetic"]
        
        context = {
            "domain": "number_theory",
            "definitions": ["prime number", "composite number"],
            "previous_results": ["Euclid's lemma"]
        }
        
        pipeline.llm_client.chat.completions.create.return_value.choices[0].message.content = \
            theorem_data["lean4"]
        pipeline.proof_assistant.verify_proof.return_value = {
            "success": True,
            "errors": [],
            "warnings": []
        }
        
        result = pipeline.formalize(theorem_data["latex"], context=context)
        
        assert result["success"] == True
        # Verify context was used in the generation process
        call_args = pipeline.llm_client.chat.completions.create.call_args
        prompt = str(call_args)
        assert "number_theory" in prompt.lower()


@pytest.mark.integration
class TestCrossSystemTranslation:
    """Test translation between different proof assistant systems."""
    
    def test_lean_to_isabelle_translation(self, mock_llm_client):
        """Test translating Lean proof to Isabelle."""
        from autoformalize.translation.cross_system import CrossSystemTranslator
        
        translator = CrossSystemTranslator(llm_client=mock_llm_client)
        lean_proof = ALL_THEOREMS["quadratic_formula"]["lean4"]
        
        isabelle_proof = """
theorem quadratic_formula:
  fixes a b c x :: real
  assumes "a ≠ 0"  
  shows "a * x^2 + b * x + c = 0 ⟷ 
         x = (-b + sqrt(b^2 - 4*a*c)) / (2*a) ∨
         x = (-b - sqrt(b^2 - 4*a*c)) / (2*a)"
  sorry
"""
        
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
            isabelle_proof
        
        result = translator.translate(lean_proof, source="lean4", target="isabelle")
        
        assert result is not None
        assert "theorem" in result
        assert "fixes" in result  # Isabelle-specific syntax
    
    def test_translation_preserves_meaning(self, mock_llm_client):
        """Test that translations preserve mathematical meaning."""
        from autoformalize.translation.cross_system import CrossSystemTranslator
        
        translator = CrossSystemTranslator(llm_client=mock_llm_client)
        
        # Test translation chain: Lean -> Coq -> Isabelle
        original_lean = ALL_THEOREMS["euclid_infinitude_primes"]["lean4"]
        
        mock_coq_response = """
Theorem euclid_infinitude_primes : forall n : nat, exists p, p > n /\ prime p.
Proof.
  admit.
Qed.
"""
        
        mock_isabelle_response = """
theorem euclid_infinitude_primes: "∀n. ∃p>n. prime p"
  sorry
"""
        
        mock_llm_client.chat.completions.create.side_effect = [
            Mock(choices=[Mock(message=Mock(content=mock_coq_response))]),
            Mock(choices=[Mock(message=Mock(content=mock_isabelle_response))])
        ]
        
        coq_result = translator.translate(original_lean, "lean4", "coq")
        isabelle_result = translator.translate(coq_result, "coq", "isabelle")
        
        # Check that key mathematical concepts are preserved
        assert "prime" in coq_result.lower()
        assert "prime" in isabelle_result.lower()
        assert "forall" in coq_result or "∀" in coq_result
        assert "∀" in isabelle_result or "forall" in isabelle_result.lower()


@pytest.mark.integration
class TestFileProcessing:
    """Test processing of actual files and documents."""
    
    def test_process_latex_file(self, pipeline, temp_dir):
        """Test processing a complete LaTeX file."""
        # Create a sample LaTeX file
        latex_content = f"""
\\documentclass{{article}}
\\begin{{document}}

{ALL_THEOREMS["quadratic_formula"]["latex"]}

{ALL_THEOREMS["binomial_theorem"]["latex"]}

\\end{{document}}
"""
        
        latex_file = temp_dir / "sample.tex"
        latex_file.write_text(latex_content)
        
        # Mock successful processing
        pipeline.llm_client.chat.completions.create.return_value.choices[0].message.content = \
            "theorem processed : True := trivial"
        pipeline.proof_assistant.verify_proof.return_value = {
            "success": True,
            "errors": [],
            "warnings": []
        }
        
        result = pipeline.process_file(latex_file)
        
        assert result is not None
        assert len(result["theorems"]) >= 2
        assert all(t["success"] for t in result["theorems"])
    
    @patch('requests.get')
    def test_process_arxiv_paper(self, mock_get, pipeline):
        """Test processing an arXiv paper."""
        # Mock arXiv API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = ALL_THEOREMS["fundamental_theorem_arithmetic"]["latex"].encode()
        mock_get.return_value = mock_response
        
        # Mock successful processing
        pipeline.llm_client.chat.completions.create.return_value.choices[0].message.content = \
            ALL_THEOREMS["fundamental_theorem_arithmetic"]["lean4"]
        pipeline.proof_assistant.verify_proof.return_value = {
            "success": True,
            "errors": [],
            "warnings": []
        }
        
        result = pipeline.process_arxiv("2301.00001")
        
        assert result is not None
        assert result["success"] == True
        assert "arxiv_id" in result
        assert result["arxiv_id"] == "2301.00001"


@pytest.mark.slow
@pytest.mark.integration  
class TestPerformanceIntegration:
    """Test performance aspects of the integration."""
    
    def test_pipeline_timeout_handling(self, pipeline):
        """Test that pipeline respects timeout settings."""
        import time
        from concurrent.futures import TimeoutError
        
        # Mock a slow LLM response
        def slow_response(*args, **kwargs):
            time.sleep(2)  # Simulate slow response
            return Mock(choices=[Mock(message=Mock(content="slow response"))])
        
        pipeline.llm_client.chat.completions.create.side_effect = slow_response
        pipeline.timeout = 1  # 1 second timeout
        
        with pytest.raises(TimeoutError):
            pipeline.formalize(ALL_THEOREMS["quadratic_formula"]["latex"])
    
    def test_pipeline_memory_usage(self, pipeline):
        """Test pipeline memory usage with large inputs."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process a large document
        large_latex = ALL_THEOREMS["binomial_theorem"]["latex"] * 100
        
        pipeline.llm_client.chat.completions.create.return_value.choices[0].message.content = \
            "theorem large : True := trivial"
        pipeline.proof_assistant.verify_proof.return_value = {
            "success": True,
            "errors": [],
            "warnings": []
        }
        
        result = pipeline.formalize(large_latex)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
        assert result is not None


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test various configuration options in integration."""
    
    def test_different_llm_models(self, mock_proof_assistant):
        """Test pipeline with different LLM models."""
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]
        
        for model in models:
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_client.chat.completions.create.return_value.choices[0].message.content = \
                    "theorem test_model : True := trivial"
                mock_openai.return_value = mock_client
                
                pipeline = FormalizationPipeline(
                    target_system="lean4",
                    llm_model=model,
                    proof_assistant=mock_proof_assistant
                )
                
                result = pipeline.formalize("\\begin{theorem}Test\\end{theorem}")
                assert result["success"] == True
    
    def test_pipeline_with_caching(self, pipeline, temp_dir):
        """Test pipeline caching functionality."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        
        pipeline.cache_dir = cache_dir
        pipeline.use_cache = True
        
        theorem = ALL_THEOREMS["quadratic_formula"]["latex"]
        
        # First call should hit LLM
        pipeline.llm_client.chat.completions.create.return_value.choices[0].message.content = \
            "cached_result"
        pipeline.proof_assistant.verify_proof.return_value = {
            "success": True,
            "errors": [],
            "warnings": []
        }
        
        result1 = pipeline.formalize(theorem)
        
        # Second call should use cache
        pipeline.llm_client.reset_mock()
        result2 = pipeline.formalize(theorem)
        
        assert result1["formal_proof"] == result2["formal_proof"]
        # LLM should not be called second time due to caching
        assert pipeline.llm_client.chat.completions.create.call_count == 0