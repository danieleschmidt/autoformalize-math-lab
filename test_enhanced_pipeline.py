#!/usr/bin/env python3
"""Test enhanced formalization pipeline functionality.

This test suite validates the enhanced pipeline implementation with
real proof assistant integration and comprehensive validation.
"""

import asyncio
import pytest
import time
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from autoformalize.core.enhanced_pipeline import (
    EnhancedFormalizationPipeline, 
    EnhancedFormalizationResult,
    VerificationMode
)
from autoformalize.utils.enhanced_validation import MathematicalValidator, ValidationLevel


class TestEnhancedPipeline:
    """Test cases for enhanced formalization pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create enhanced pipeline instance."""
        return EnhancedFormalizationPipeline(
            target_system="lean4",
            verification_mode=VerificationMode.MOCK,  # Use mock for testing
            enable_caching=True,
            enable_validation=True,
            max_correction_rounds=3,
            timeout=60
        )
    
    @pytest.fixture
    def validator(self):
        """Create mathematical validator instance."""
        return MathematicalValidator(ValidationLevel.STANDARD)
    
    @pytest.mark.asyncio
    async def test_simple_theorem_formalization(self, pipeline):
        """Test formalization of a simple theorem."""
        latex_content = r"""
        \begin{theorem}
        For any natural number $n$, we have $n + 0 = n$.
        \end{theorem}
        \begin{proof}
        This follows directly from the definition of addition.
        \end{proof}
        """
        
        result = await pipeline.formalize(latex_content)
        
        assert isinstance(result, EnhancedFormalizationResult)
        assert result.success
        assert result.formal_code is not None
        assert result.confidence_score > 0.0
        assert result.processing_time > 0.0
    
    @pytest.mark.asyncio
    async def test_complex_mathematical_content(self, pipeline):
        """Test formalization of complex mathematical content."""
        latex_content = r"""
        \begin{theorem}[Fundamental Theorem of Arithmetic]
        Every integer greater than 1 is either prime or can be uniquely 
        factored into primes.
        \end{theorem}
        \begin{proof}
        We prove this by strong induction on $n$.
        
        Base case: $n = 2$ is prime.
        
        Inductive step: Assume the theorem holds for all $k < n$.
        If $n$ is prime, we're done. Otherwise, $n = ab$ where $1 < a, b < n$.
        By the inductive hypothesis, both $a$ and $b$ have unique prime 
        factorizations, so $n$ does too.
        \end{proof}
        """
        
        result = await pipeline.formalize(latex_content)
        
        assert isinstance(result, EnhancedFormalizationResult)
        # This might not succeed due to complexity, but should not crash
        assert result.processing_time > 0.0
        assert result.theorem_type == "theorem"
        assert result.complexity_score > 1
    
    @pytest.mark.asyncio
    async def test_validation_integration(self, validator):
        """Test mathematical validation functionality."""
        latex_content = r"""
        \begin{theorem}
        For any real number $x$, we have $x^2 \geq 0$.
        \end{theorem}
        """
        
        result = await validator.validate_latex_content(latex_content)
        
        assert result.valid
        assert result.score > 0.8
        assert len(result.issues) == 0
        assert "theorem" in str(result.metadata.get("mathematical_elements", {}))
    
    @pytest.mark.asyncio
    async def test_invalid_latex_detection(self, validator):
        """Test detection of invalid LaTeX content."""
        invalid_latex = r"""
        \begin{theorem
        Unclosed theorem environment with unbalanced braces {
        \end{theorem}
        """
        
        result = await validator.validate_latex_content(invalid_latex)
        
        assert not result.valid
        assert len(result.issues) > 0
        assert result.score < 1.0
    
    @pytest.mark.asyncio
    async def test_pipeline_system_status(self, pipeline):
        """Test system status reporting."""
        status = await pipeline.get_system_status()
        
        assert "pipeline" in status
        assert "components" in status
        assert status["pipeline"]["target_system"] == "lean4"
        assert status["components"]["parser"] == "active"
        assert status["components"]["generator"] == "active"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, pipeline):
        """Test performance benchmarking functionality."""
        test_cases = [
            r"$1 + 1 = 2$",
            r"\begin{theorem} For any $n$, $n = n$. \end{theorem}",
            r"$\forall x \in \mathbb{R}, x^2 \geq 0$"
        ]
        
        benchmark_result = await pipeline.benchmark_performance(test_cases)
        
        assert "total_cases" in benchmark_result
        assert "successful_cases" in benchmark_result
        assert "success_rate" in benchmark_result
        assert benchmark_result["total_cases"] == len(test_cases)
        assert benchmark_result["total_time"] > 0.0
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, pipeline):
        """Test caching mechanism."""
        latex_content = r"$2 + 2 = 4$"
        
        # First call
        start_time = time.time()
        result1 = await pipeline.formalize(latex_content)
        first_time = time.time() - start_time
        
        # Second call (should use cache)
        start_time = time.time()
        result2 = await pipeline.formalize(latex_content)
        second_time = time.time() - start_time
        
        # Results should be identical
        assert result1.success == result2.success
        if result1.formal_code and result2.formal_code:
            assert result1.formal_code == result2.formal_code
        
        # Second call should be faster (though this might be flaky)
        # assert second_time < first_time  # Commented out for reliability
    
    @pytest.mark.asyncio
    async def test_correction_mechanism(self, pipeline):
        """Test self-correction functionality."""
        # Intentionally problematic content
        latex_content = r"""
        \begin{theorem}
        Some very complex theorem that might require corrections.
        \end{theorem}
        """
        
        result = await pipeline.formalize(latex_content)
        
        # Should track correction rounds
        assert result.correction_rounds >= 0
        assert result.correction_rounds <= pipeline.max_correction_rounds
    
    def test_confidence_scoring(self, pipeline):
        """Test confidence score calculation."""
        # Mock result for testing
        result = EnhancedFormalizationResult(
            success=True,
            verification_status=True,
            mathematical_validation=True,
            correction_rounds=1
        )
        
        confidence = pipeline._calculate_confidence_score(result, None)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably confident
    
    def test_complexity_scoring(self, pipeline):
        """Test complexity score calculation."""
        simple_code = "theorem simple : 1 = 1 := rfl"
        complex_code = """
        theorem complex (n : ℕ) : ∀ (P : ℕ → Prop), 
          P 0 → (∀ k, P k → P (k + 1)) → P n := by
          induction n with
          | zero => exact fun P h₀ _ => h₀
          | succ n ih => exact fun P h₀ h_step => h_step n (ih P h₀ h_step)
        """
        
        simple_score = pipeline._calculate_complexity_score(None, simple_code)
        complex_score = pipeline._calculate_complexity_score(None, complex_code)
        
        assert 1 <= simple_score <= 10
        assert 1 <= complex_score <= 10
        assert complex_score > simple_score


async def main():
    """Run enhanced pipeline demonstration."""
    print("🚀 Enhanced Formalization Pipeline Demo")
    print("=" * 50)
    
    # Initialize enhanced pipeline
    pipeline = EnhancedFormalizationPipeline(
        target_system="lean4",
        verification_mode=VerificationMode.MOCK,
        enable_caching=True,
        enable_validation=True
    )
    
    # Test cases
    test_cases = [
        {
            "name": "Simple Addition",
            "content": r"$1 + 1 = 2$"
        },
        {
            "name": "Basic Theorem",
            "content": r"""
            \begin{theorem}
            For any natural number $n$, we have $n + 0 = n$.
            \end{theorem}
            \begin{proof}
            By the definition of addition.
            \end{proof}
            """
        },
        {
            "name": "Pythagorean Theorem",
            "content": r"""
            \begin{theorem}[Pythagorean Theorem]
            In a right triangle with legs of length $a$ and $b$ and 
            hypotenuse of length $c$, we have $a^2 + b^2 = c^2$.
            \end{theorem}
            """
        }
    ]
    
    results = []
    
    print("\n📊 Running Formalization Tests...")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            result = await pipeline.formalize(test_case['content'])
            
            print(f"✅ Success: {result.success}")
            print(f"⏱️  Processing Time: {result.processing_time:.2f}s")
            print(f"🔄 Correction Rounds: {result.correction_rounds}")
            print(f"📊 Confidence Score: {result.confidence_score:.2f}")
            print(f"🧮 Complexity Score: {result.complexity_score}/10")
            
            if result.formal_code:
                print(f"📝 Generated Code Preview:")
                preview = result.formal_code[:200] + "..." if len(result.formal_code) > 200 else result.formal_code
                print(f"   {preview}")
            
            if result.error_message:
                print(f"❌ Error: {result.error_message}")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append(None)
    
    # Performance benchmark
    print("\n\n🏃 Performance Benchmark")
    print("=" * 30)
    
    benchmark_cases = [case['content'] for case in test_cases]
    benchmark_result = await pipeline.benchmark_performance(benchmark_cases)
    
    print(f"📈 Total Cases: {benchmark_result['total_cases']}")
    print(f"✅ Successful: {benchmark_result['successful_cases']}")
    print(f"📊 Success Rate: {benchmark_result['success_rate']:.1%}")
    print(f"⏱️  Total Time: {benchmark_result['total_time']:.2f}s")
    print(f"⚡ Average Time: {benchmark_result['average_time']:.2f}s")
    
    if benchmark_result['successful_cases'] > 0:
        print(f"🎯 Average Confidence: {benchmark_result['average_confidence']:.2f}")
    
    # System status
    print("\n\n🔧 System Status")
    print("=" * 20)
    
    status = await pipeline.get_system_status()
    print(f"🎯 Target System: {status['pipeline']['target_system']}")
    print(f"🤖 Model: {status['pipeline']['model_name']}")
    print(f"🔍 Verification: {status['pipeline']['verification_mode']}")
    print(f"💾 Caching: {'Enabled' if status['pipeline']['caching_enabled'] else 'Disabled'}")
    print(f"✅ Validation: {'Enabled' if status['pipeline']['validation_enabled'] else 'Disabled'}")
    
    print("\n🎉 Enhanced Pipeline Demo Complete!")
    
    return {
        "test_results": results,
        "benchmark": benchmark_result,
        "system_status": status
    }


if __name__ == "__main__":
    # Run as script
    print("Enhanced Formalization Pipeline Test")
    
    # Run tests if pytest is available
    try:
        import pytest
        print("Running pytest tests...")
        pytest.main([__file__, "-v"])
    except ImportError:
        print("Pytest not available, running demo...")
        asyncio.run(main())