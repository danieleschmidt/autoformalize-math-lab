"""Unit tests for formal proof generators."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from autoformalize.generators.lean import Lean4Generator
from autoformalize.generators.isabelle import IsabelleGenerator
from autoformalize.generators.coq import CoqGenerator
from autoformalize.core.correction import CorrectionEngine
from tests.fixtures import ALL_THEOREMS, get_sample_proof_assistant_errors


class TestLean4Generator:
    """Test cases for Lean 4 proof generation."""
    
    def test_generate_basic_theorem(self, mock_llm_client):
        """Test generation of basic Lean 4 theorem."""
        generator = Lean4Generator(llm_client=mock_llm_client)
        theorem_data = ALL_THEOREMS["quadratic_formula"]
        
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
            theorem_data["lean4"]
        
        result = generator.generate(theorem_data["latex"])
        
        assert result is not None
        assert "theorem" in result.lower()
        assert "quadratic_formula" in result or "quadratic" in result.lower()
    
    def test_generate_with_mathlib_imports(self, mock_llm_client):
        """Test generation with appropriate Mathlib imports.""" 
        generator = Lean4Generator(llm_client=mock_llm_client, use_mathlib=True)
        theorem_data = ALL_THEOREMS["fundamental_theorem_arithmetic"]
        
        mock_response = f"""
import Mathlib.NumberTheory.PrimeCounting
import Mathlib.Data.Nat.Prime

{theorem_data["lean4"]}
"""
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = mock_response
        
        result = generator.generate(theorem_data["latex"])
        
        assert "import Mathlib" in result
        assert "Nat.Prime" in result
    
    def test_generate_with_context(self, mock_llm_client):
        """Test generation with additional mathematical context."""
        generator = Lean4Generator(llm_client=mock_llm_client)
        
        context = {
            "definitions": ["Prime number", "Composite number"],
            "previous_theorems": ["Euclid's lemma"],
            "domain": "number_theory"
        }
        
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
            "theorem with_context : ℕ → Prop := sorry"
        
        result = generator.generate("Sample theorem", context=context)
        
        assert result is not None
        # Verify that context was used in the prompt
        call_args = mock_llm_client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][-1]["content"]
        assert "number_theory" in prompt.lower()
    
    def test_error_handling(self, mock_llm_client):
        """Test handling of LLM API errors."""
        generator = Lean4Generator(llm_client=mock_llm_client)
        
        # Simulate API error
        mock_llm_client.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            generator.generate("Sample theorem")
    
    def test_validate_lean_syntax(self):
        """Test Lean syntax validation."""
        generator = Lean4Generator()
        
        valid_lean = "theorem test : 1 + 1 = 2 := by norm_num"
        invalid_lean = "theorem test : 1 + 1 = 2 := by invalid_tactic"
        
        assert generator.validate_syntax(valid_lean) == True
        assert generator.validate_syntax(invalid_lean) == False


class TestIsabelleGenerator:
    """Test cases for Isabelle/HOL proof generation."""
    
    def test_generate_basic_theorem(self, mock_llm_client):
        """Test generation of basic Isabelle theorem."""
        generator = IsabelleGenerator(llm_client=mock_llm_client)
        
        mock_response = """
theory Test
imports Main
begin

theorem quadratic_formula:
  fixes a b c x :: real
  assumes "a ≠ 0"
  shows "a * x^2 + b * x + c = 0 ⟷ 
         x = (-b + sqrt(b^2 - 4*a*c)) / (2*a) ∨ 
         x = (-b - sqrt(b^2 - 4*a*c)) / (2*a)"
  sorry

end
"""
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = mock_response
        
        result = generator.generate(ALL_THEOREMS["quadratic_formula"]["latex"])
        
        assert result is not None
        assert "theory" in result
        assert "theorem" in result
        assert "imports Main" in result
    
    def test_generate_with_proof_methods(self, mock_llm_client):
        """Test generation using Isabelle proof methods."""
        generator = IsabelleGenerator(llm_client=mock_llm_client)
        
        mock_response = """
theorem simple_theorem: "1 + 1 = (2::nat)"
  by simp
"""
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = mock_response
        
        result = generator.generate("Theorem: 1 + 1 = 2")
        
        assert "by simp" in result or "by auto" in result


class TestCoqGenerator:
    """Test cases for Coq proof generation."""
    
    def test_generate_basic_theorem(self, mock_llm_client):
        """Test generation of basic Coq theorem."""
        generator = CoqGenerator(llm_client=mock_llm_client)
        
        mock_response = """
Require Import Arith.

Theorem quadratic_formula : forall a b c x : R,
  a <> 0 ->
  a * x^2 + b * x + c = 0 <->
  x = (-b + sqrt(b^2 - 4*a*c)) / (2*a) \/
  x = (-b - sqrt(b^2 - 4*a*c)) / (2*a).
Proof.
  intros.
  admit.
Qed.
"""
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = mock_response
        
        result = generator.generate(ALL_THEOREMS["quadratic_formula"]["latex"])
        
        assert result is not None
        assert "Theorem" in result
        assert "Proof." in result
        assert "Qed." in result
    
    def test_generate_with_tactics(self, mock_llm_client):
        """Test generation using Coq tactics."""
        generator = CoqGenerator(llm_client=mock_llm_client)
        
        mock_response = """
Theorem simple : 1 + 1 = 2.
Proof.
  reflexivity.
Qed.
"""
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = mock_response
        
        result = generator.generate("Theorem: 1 + 1 = 2")
        
        assert "reflexivity" in result or "auto" in result


class TestCorrectionEngine:
    """Test cases for self-correction functionality."""
    
    def test_parse_lean_errors(self):
        """Test parsing of Lean error messages."""
        engine = CorrectionEngine()
        errors = get_sample_proof_assistant_errors()["lean4"]
        
        for error_type, error_msg in errors.items():
            parsed = engine.parse_error(error_msg, "lean4")
            
            assert parsed is not None
            assert "type" in parsed
            assert "message" in parsed
            assert "suggestion" in parsed
    
    def test_generate_correction_prompt(self, mock_llm_client):
        """Test generation of correction prompts."""
        engine = CorrectionEngine(llm_client=mock_llm_client)
        
        original_proof = "theorem test : 1 + 1 = 2 := by unknown_tactic"
        error_msg = "unknown tactic: unknown_tactic"
        
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
            "theorem test : 1 + 1 = 2 := by norm_num"
        
        corrected = engine.generate_correction(original_proof, error_msg, "lean4")
        
        assert corrected is not None
        assert "norm_num" in corrected or "simp" in corrected
    
    def test_correction_iteration_limit(self, mock_llm_client):
        """Test that correction engine respects iteration limits."""
        engine = CorrectionEngine(llm_client=mock_llm_client, max_corrections=3)
        
        # Mock persistent error
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
            "theorem test : 1 + 1 = 2 := by still_wrong"
        
        original_proof = "theorem test : 1 + 1 = 2 := by wrong"
        error_msg = "tactic failed"
        
        corrections = []
        for i in range(5):  # Try more than the limit
            try:
                corrected = engine.generate_correction(original_proof, error_msg, "lean4")
                corrections.append(corrected)
            except Exception as e:
                if "maximum corrections" in str(e).lower():
                    break
        
        assert len(corrections) <= 3
    
    @pytest.mark.slow
    def test_full_correction_loop(self, mock_llm_client, mock_proof_assistant):
        """Test complete self-correction loop.""" 
        engine = CorrectionEngine(
            llm_client=mock_llm_client,
            proof_assistant=mock_proof_assistant
        )
        
        # First attempt fails
        mock_proof_assistant.verify_proof.side_effect = [
            {"success": False, "errors": ["syntax error"], "warnings": []},
            {"success": True, "errors": [], "warnings": []}
        ]
        
        # Correction succeeds
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
            "theorem test : 1 + 1 = 2 := by norm_num"
        
        original = "theorem test : 1 + 1 = 2 := by wrong"
        result = engine.self_correct(original, "lean4")
        
        assert result["success"] == True
        assert result["final_proof"] is not None
        assert result["correction_rounds"] == 1


@pytest.mark.mathematical
class TestMathematicalAccuracy:
    """Test mathematical accuracy of generated proofs."""
    
    def test_preserve_mathematical_meaning(self, mock_llm_client):
        """Test that generated proofs preserve mathematical meaning."""
        generator = Lean4Generator(llm_client=mock_llm_client)
        
        theorem_data = ALL_THEOREMS["fundamental_theorem_arithmetic"]
        
        # Mock a mathematically accurate response
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
            theorem_data["lean4"]
        
        result = generator.generate(theorem_data["latex"])
        
        # Check key mathematical concepts are preserved
        assert "prime" in result.lower()
        assert "unique" in result.lower() or "∃!" in result
        assert "multiset" in result.lower() or "factors" in result.lower()
    
    def test_domain_specific_generation(self, mock_llm_client):
        """Test generation quality varies by mathematical domain."""
        generator = Lean4Generator(llm_client=mock_llm_client)
        
        domains = ["algebra", "number_theory", "analysis"]
        results = {}
        
        for domain in domains:
            theorems = [t for t in ALL_THEOREMS.values() if t["domain"] == domain]
            if theorems:
                theorem = theorems[0]
                mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
                    theorem["lean4"]
                
                result = generator.generate(theorem["latex"])
                results[domain] = result
        
        # Each domain should produce different characteristic patterns
        assert len(results) > 0
        for domain, result in results.items():
            assert result is not None
            assert len(result.strip()) > 0